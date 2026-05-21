import os
import warnings
import torch
import numpy as np
import pandas as pd
import option
import yaml
from torch import nn

# NNI is optional (for hyperparameter tuning)
try:
    import nni
    from nni.utils import merge_parameter
    NNI_AVAILABLE = True
except ImportError:
    NNI_AVAILABLE = False
    print("⚠️  NNI not installed, running without hyperparameter tuning")

import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

from models.EAMBNet_AGREE import EAMBNet_AGREE
from multimodal_dataset import MultimodalPARADataset
from util import AverageMeter


def load_config(config_path='config.yml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# 全局配置
SAGA_CONFIG = load_config()
SAGA_VERSION = SAGA_CONFIG.get('saga_version', 'v2')
SENSITIVITY_CSV_PATH = SAGA_CONFIG.get('sensitivity_csv_path', None)
SENSITIVITY_ALPHA = SAGA_CONFIG.get('sensitivity_alpha', 0.2)
CONFIG_CKPT_DIR = SAGA_CONFIG.get('checkpoint_dir', None)

# AGREE：误差感知加权配置（不引入HMAI指标）
ERROR_AWARE_CONFIG = SAGA_CONFIG.get('error_aware', {})
ERROR_AWARE_ENABLED = ERROR_AWARE_CONFIG.get('enabled', False)
ERROR_AWARE_BETA = ERROR_AWARE_CONFIG.get('beta', 1.0)
ERROR_AWARE_TAU = ERROR_AWARE_CONFIG.get('tau', 0.1)
ERROR_AWARE_WARMUP = ERROR_AWARE_CONFIG.get('warmup_epochs', 5)
ERROR_AWARE_EMA_DECAY = ERROR_AWARE_CONFIG.get('ema_decay', 0.9)

print(f"\n{'='*80}")
print(f"🚀 EAMB-Net AGREE训练脚本 - PARA (不含HMAI指标)")
print(f"   敏感度CSV: {SENSITIVITY_CSV_PATH if SENSITIVITY_CSV_PATH else '无（v1模式）'}")
if SAGA_VERSION == 'v3':
    print(f"   敏感度损失权重α: {SENSITIVITY_ALPHA}")
if ERROR_AWARE_ENABLED:
    print(f"   误差感知加权: ✅ 启用 (权重归一化mean=1, warmup={ERROR_AWARE_WARMUP})")
else:
    print(f"   误差感知加权: ❌ 未启用")
if CONFIG_CKPT_DIR:
    print(f"   checkpoint_dir: {CONFIG_CKPT_DIR}")
print(f"{'='*80}\n")


# ========== 多任务学习：敏感度数据加载（仅v3使用）==========
sensitivity_data = None


def load_sensitivity_data():
    global sensitivity_data
    if SENSITIVITY_CSV_PATH and sensitivity_data is None:
        sensitivity_data = pd.read_csv(SENSITIVITY_CSV_PATH)
        sensitivity_data.set_index('image_id', inplace=True)
        print(f"✅ 多任务学习：加载敏感度数据 {len(sensitivity_data)} 条")
    return sensitivity_data


def get_sensitivity_batch(image_ids, device):
    if SAGA_VERSION != 'v3':
        return None
    sens_data = load_sensitivity_data()
    if sens_data is None:
        return None

    batch_size = len(image_ids)
    sensitivity_weights = torch.zeros(batch_size, 5)
    attr_cols = ['sens_brightness', 'sens_contrast', 'sens_blur', 'sens_hue', 'sens_saturation']
    for i, img_id in enumerate(image_ids):
        try:
            row = sens_data.loc[img_id]
            sens_values = [row[col] for col in attr_cols]
            sens_tensor = torch.tensor(sens_values, dtype=torch.float32)
            sens_tensor = torch.clamp(sens_tensor, min=0.0)
            sensitivity_weights[i] = sens_tensor / (sens_tensor.sum() + 1e-8)
        except KeyError:
            sensitivity_weights[i] = torch.ones(5) / 5.0
    return sensitivity_weights.to(device)


class ErrorTrackerV2:
    """误差追踪器：困难样本加权，权重归一化 mean=1（不引入HMAI指标）"""
    def __init__(self, ema_decay=0.9):
        self.ema_decay = ema_decay
        self.error_history = {}
        self.global_mean = 0.0
        self.global_std = 1.0
        self.global_ema_decay = 0.99

    def update(self, image_ids, errors):
        batch_errors = []
        for img_id, err in zip(image_ids, errors):
            err_val = err.item() if torch.is_tensor(err) else float(err)
            batch_errors.append(err_val)
            if img_id in self.error_history:
                self.error_history[img_id] = self.ema_decay * self.error_history[img_id] + (1 - self.ema_decay) * err_val
            else:
                self.error_history[img_id] = err_val

        if len(batch_errors) > 1:
            batch_mean = np.mean(batch_errors)
            batch_std = np.std(batch_errors) + 1e-6
            if self.global_mean == 0.0:
                self.global_mean = batch_mean
                self.global_std = batch_std
            else:
                self.global_mean = self.global_ema_decay * self.global_mean + (1 - self.global_ema_decay) * batch_mean
                self.global_std = self.global_ema_decay * self.global_std + (1 - self.global_ema_decay) * batch_std

    def get_weights(self, image_ids, errors, beta=1.0, tau=0.1):
        hist = []
        for img_id, err in zip(image_ids, errors):
            err_val = err.item() if torch.is_tensor(err) else float(err)
            hist.append(self.error_history.get(img_id, err_val))
        hist = torch.tensor(hist, device=errors.device, dtype=torch.float32)

        rel = (hist - self.global_mean) / (tau * self.global_std + 1e-6)
        extra = torch.clamp(rel, min=0.0)
        weights = 1.0 + beta * torch.tanh(extra)
        if weights.numel() > 0:
            weights = weights / (weights.mean() + 1e-12)
        return weights

    def get_stats(self):
        if not self.error_history:
            return 0.0, 0.0, 0
        errors = list(self.error_history.values())
        return float(np.mean(errors)), float(np.std(errors)), len(errors)


error_tracker = ErrorTrackerV2(ema_decay=ERROR_AWARE_EMA_DECAY) if ERROR_AWARE_ENABLED else None


def create_data_part(opt):
    csv_root = opt['path_to_save_csv']
    images_path = opt['path_to_images']
    text_root = opt['path_to_text_features']

    train_ds = MultimodalPARADataset(os.path.join(csv_root, 'train.csv'), images_path, text_root, if_train=True)
    val_ds = MultimodalPARADataset(os.path.join(csv_root, 'val.csv'), images_path, text_root, if_train=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    return train_loader, val_loader


def _freeze_emotion_model(model):
    for param in model.emotion_model.parameters():
        param.requires_grad = False


def _prepare_batch(batch, device):
    (x, y,
     brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr,
     overall_text, brightness_text, contrast_text, saturation_text, hue_text, blur_text,
     image_ids) = batch

    tensors = [brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr]
    text_tensors = [overall_text, brightness_text, contrast_text, saturation_text, hue_text, blur_text]

    x = x.to(device)
    y = y.to(device).view(y.size(0), -1).float()
    attr_tensors = [t.to(device) for t in tensors]
    text_tensors = [t.to(device).float() for t in text_tensors]
    return x, y, attr_tensors, text_tensors, image_ids


def train(opt, model, loader, optimizer, criterion, device, epoch=0):
    model.train()
    _freeze_emotion_model(model)
    train_losses = AverageMeter()
    score_losses = AverageMeter()
    sens_losses = AverageMeter()
    error_weights = AverageMeter()

    use_error_aware = (
        ERROR_AWARE_ENABLED and
        epoch >= ERROR_AWARE_WARMUP and
        error_tracker is not None and
        SAGA_VERSION != 'v3'
    )
    if use_error_aware:
        print(f"   🎯 AGREE误差感知加权已启用 (epoch {epoch+1} > warmup {ERROR_AWARE_WARMUP})")

    for batch in tqdm(loader):
        x, y, attr_tensors, text_tensors, image_ids = _prepare_batch(batch, device)
        (brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr) = attr_tensors
        (overall_text, brightness_text, contrast_text, saturation_text, hue_text, blur_text) = text_tensors

        if SAGA_VERSION == 'v3':
            y_pred, pred_sensitivity = model(
                x,
                brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr,
                overall_text, brightness_text, contrast_text, saturation_text, hue_text, blur_text,
                image_ids=image_ids,
                return_sensitivity=True
            )
            score_loss = criterion(y_pred, y)

            true_sensitivity = get_sensitivity_batch(image_ids, device)
            if true_sensitivity is not None:
                sens_loss = F.kl_div(
                    torch.log(pred_sensitivity + 1e-10),
                    true_sensitivity,
                    reduction='batchmean'
                )
                total_loss = score_loss + SENSITIVITY_ALPHA * sens_loss
                sens_losses.update(sens_loss.item(), x.size(0))
            else:
                total_loss = score_loss
        else:
            y_pred = model(
                x,
                brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr,
                overall_text, brightness_text, contrast_text, saturation_text, hue_text, blur_text,
                image_ids=image_ids
            )

            if use_error_aware:
                sample_errors = torch.abs(y_pred - y).view(-1)
                weights = error_tracker.get_weights(
                    image_ids, sample_errors.detach(),
                    beta=ERROR_AWARE_BETA,
                    tau=ERROR_AWARE_TAU
                )
                sample_losses = (y_pred - y).pow(2).view(-1)
                score_loss = (weights * sample_losses).mean()

                error_tracker.update(image_ids, sample_errors.detach())
                error_weights.update(weights.mean().item(), x.size(0))
            else:
                score_loss = criterion(y_pred, y)
                if ERROR_AWARE_ENABLED and error_tracker is not None:
                    sample_errors = torch.abs(y_pred - y).view(-1).detach()
                    error_tracker.update(image_ids, sample_errors)

            total_loss = score_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_losses.update(total_loss.item(), x.size(0))
        score_losses.update(score_loss.item(), x.size(0))

    if SAGA_VERSION == 'v3':
        print(f"   Train - Total: {train_losses.avg:.6f}, Score: {score_losses.avg:.6f}, Sens: {sens_losses.avg:.6f}")
    else:
        if ERROR_AWARE_ENABLED and error_tracker is not None:
            mean_err, std_err, n_samples = error_tracker.get_stats()
            if use_error_aware:
                print(f"   Train - Loss: {train_losses.avg:.6f}, AvgWeight: {error_weights.avg:.3f}")
            else:
                print(f"   Train - Loss: {train_losses.avg:.6f} (预热中, epoch {epoch+1}/{ERROR_AWARE_WARMUP})")
            print(f"   Error Stats - GlobalMean: {error_tracker.global_mean:.4f}, GlobalStd: {error_tracker.global_std:.4f}, Tracked: {n_samples}")
        else:
            print(f"   Train - Loss: {train_losses.avg:.6f}")

    return train_losses.avg


def validate(opt, model, loader, criterion, device):
    model.eval()
    validate_losses = AverageMeter()
    true_scores, pred_scores = [], []

    with torch.no_grad():
        for batch in tqdm(loader):
            x, y, attr_tensors, text_tensors, image_ids = _prepare_batch(batch, device)
            (brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr) = attr_tensors
            (overall_text, brightness_text, contrast_text, saturation_text, hue_text, blur_text) = text_tensors

            if SAGA_VERSION == 'v3':
                y_pred, _ = model(
                    x,
                    brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr,
                    overall_text, brightness_text, contrast_text, saturation_text, hue_text, blur_text,
                    image_ids=image_ids,
                    return_sensitivity=True
                )
            else:
                y_pred = model(
                    x,
                    brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr,
                    overall_text, brightness_text, contrast_text, saturation_text, hue_text, blur_text,
                    image_ids=image_ids
                )

            loss = criterion(y_pred, y)
            validate_losses.update(loss.item(), x.size(0))

            pred_scores += y_pred.data.cpu().numpy().astype('float').mean(axis=1).tolist()
            true_scores += y.data.cpu().numpy().astype('float').mean(axis=1).tolist()

    srcc_mean, _ = spearmanr(pred_scores, true_scores)
    lcc_mean, _ = pearsonr(pred_scores, true_scores)
    mae = mean_absolute_error(true_scores, pred_scores)
    mse = mean_squared_error(true_scores, pred_scores)
    rmse = np.sqrt(mse)

    true_scores = np.array(true_scores)
    pred_scores = np.array(pred_scores)
    true_label = np.where(true_scores <= 0.55, 0, 1)
    pred_label = np.where(pred_scores <= 0.55, 0, 1)
    acc = accuracy_score(true_label, pred_label)

    print('accuracy: {:.4f}, lcc_mean: {:.4f}, srcc_mean: {:.4f}, mae: {:.4f}, mse: {:.4f}, rmse: {:.4f}'.format(
        acc, lcc_mean, srcc_mean, mae, mse, rmse))
    return validate_losses.avg, acc, srcc_mean, lcc_mean


def start_train(opt):
    device = torch.device(f"cuda:{opt['gpu_id']}")
    train_loader, val_loader = create_data_part(opt)

    model = EAMBNet_AGREE(
        saga_version=SAGA_VERSION,
        sensitivity_csv_path=SENSITIVITY_CSV_PATH
    ).to(device)

    criterion = nn.MSELoss().to(device)

    attribute_params = list(model.attribute_parameters())
    fusion_params = list(model.fusion_parameters())
    optimizer = optim.Adam([
        {'params': model.emotion_model.parameters(), 'lr': opt['init_lr_emotion']},
        {'params': model.model_x.parameters(), 'lr': opt['init_lr_visual']},
        {'params': model.model_s.parameters(), 'lr': opt['init_lr_visual']},
        {'params': attribute_params, 'lr': opt['init_lr_visual']},
        {'params': fusion_params, 'lr': opt['init_lr_fusion']},
        {'params': model.head.parameters(), 'lr': opt['init_lr_head']},
    ], lr=opt['init_lr'])

    resume_path = opt.get('resume', '')
    start_epoch = 0
    best_srcc = -1e9
    best_acc = -1e9

    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0)
        best_srcc = checkpoint.get('srcc_val', best_srcc)
        best_acc = checkpoint.get('vacc', best_acc)
        print(f"🔄 恢复训练: epoch {start_epoch}, best SRCC {best_srcc:.4f}, best ACC {best_acc:.4f}")
    elif resume_path:
        print(f"⚠️  未找到指定 checkpoint: {resume_path}")

    for epoch in range(start_epoch, opt['num_epoch']):
        print(f"\n===== Epoch {epoch + 1}/{opt['num_epoch']} =====")
        train_loss = train(opt, model, train_loader, optimizer, criterion, device, epoch=epoch)
        val_loss, vacc, vsrcc, vlcc = validate(opt, model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{opt['num_epoch']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        save_dir = CONFIG_CKPT_DIR or opt["path_to_save_ckpt"]
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'epoch_{epoch+1:03d}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'srcc_val': vsrcc,
            'plcc_val': vlcc,
            'vacc': vacc
        }, save_path)
        print(f"✅ Saved model to {save_path}")

        best_srcc = max(best_srcc, vsrcc)
        best_acc = max(best_acc, vacc)

        if NNI_AVAILABLE:
            nni.report_intermediate_result({'default': vacc, "vsrcc": vsrcc, "val_loss": val_loss})

    if NNI_AVAILABLE:
        nni.report_final_result({'default': best_acc, "vsrcc": best_srcc})


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    base_opt = option.init()
    if NNI_AVAILABLE:
        tuner_params = nni.get_next_parameter()
        merged_opt = vars(merge_parameter(base_opt, tuner_params))
    else:
        merged_opt = vars(base_opt)
    start_train(merged_opt)

