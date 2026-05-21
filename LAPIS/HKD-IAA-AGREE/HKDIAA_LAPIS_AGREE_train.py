import os
import warnings
from tqdm import tqdm
import torch
import numpy as np
import yaml
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

import option

# NNI is optional (for hyperparameter tuning)
try:
    import nni
    from nni.utils import merge_parameter
    NNI_AVAILABLE = True
except ImportError:
    NNI_AVAILABLE = False
    print("⚠️  NNI not installed, running without hyperparameter tuning")

from models.HKD_Fusion import HKDFusion
from multimodal_dataset import MultimodalPARADataset
from util import AverageMeter

warnings.filterwarnings('ignore')


def load_config(config_path='config.yml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# 全局配置
SAGA_CONFIG = load_config()
SAGA_VERSION = SAGA_CONFIG.get('saga_version', 'v2')
SENSITIVITY_CSV_PATH = SAGA_CONFIG.get('sensitivity_csv_path', None)
CONFIG_CKPT_DIR = SAGA_CONFIG.get('checkpoint_dir', None)

# AGREE：误差感知加权配置（不引入HMAI指标）
ERROR_AWARE_CONFIG = SAGA_CONFIG.get('error_aware', {})
ERROR_AWARE_ENABLED = ERROR_AWARE_CONFIG.get('enabled', False)
ERROR_AWARE_BETA = ERROR_AWARE_CONFIG.get('beta', 1.0)
ERROR_AWARE_TAU = ERROR_AWARE_CONFIG.get('tau', 0.1)
ERROR_AWARE_WARMUP = ERROR_AWARE_CONFIG.get('warmup_epochs', 5)
ERROR_AWARE_EMA_DECAY = ERROR_AWARE_CONFIG.get('ema_decay', 0.9)

print(f"\n{'='*80}")
print(f"🚀 HKD-IAA AGREE训练脚本 - LAPIS (不含HMAI指标)")
print(f"   敏感度CSV: {SENSITIVITY_CSV_PATH if SENSITIVITY_CSV_PATH else '无（v1模式）'}")
if ERROR_AWARE_ENABLED:
    print(f"   误差感知加权: ✅ 启用 (权重归一化mean=1, warmup={ERROR_AWARE_WARMUP})")
else:
    print(f"   误差感知加权: ❌ 未启用")
if CONFIG_CKPT_DIR:
    print(f"   checkpoint_dir: {CONFIG_CKPT_DIR}")
print(f"{'='*80}\n")


def adjust_learning_rate(opt, optimizer, epoch):
    lr = opt['init_lr'] * (0.5 ** (epoch // 10))
    print(f"Epoch {epoch+1} LR : {lr:.6f}")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_data_part(opt):
    csv_root = opt['path_to_PARA_save_csv']
    images = opt['path_to_PARA_images']
    text_root = opt['path_to_text_features']

    train_ds = MultimodalPARADataset(os.path.join(csv_root, 'train.csv'), images, text_root, if_train=True)
    val_ds = MultimodalPARADataset(os.path.join(csv_root, 'val.csv'), images, text_root, if_train=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=opt['batch_size'], num_workers=opt['train_num_workers'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=opt['batch_size'], num_workers=opt['test_num_workers'], shuffle=False)
    return train_loader, val_loader


def _fix_text_feature_dim(text_feat, device):
    text_feat = text_feat.to(device).float()
    if text_feat.dim() == 3:
        text_feat = text_feat.squeeze(1)
    elif text_feat.dim() == 1:
        text_feat = text_feat.unsqueeze(0)
    return text_feat


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


def train(opt, model, loader, optimizer, criterion, device, epoch=0):
    model.train()
    train_losses = AverageMeter()
    error_weights = AverageMeter()

    use_error_aware = (
        ERROR_AWARE_ENABLED and
        epoch >= ERROR_AWARE_WARMUP and
        error_tracker is not None
    )
    if use_error_aware:
        print(f"   🎯 AGREE误差感知加权已启用 (epoch {epoch+1} > warmup {ERROR_AWARE_WARMUP})")

    for batch in tqdm(loader):
        (x, y,
         brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr,
         overall_text, brightness_text, contrast_text, saturation_text, hue_text, blur_text, image_ids) = batch

        x = x.to(device)
        y = y.to(device).view(y.size(0), -1).float()
        brightness_attr = brightness_attr.to(device)
        contrast_attr = contrast_attr.to(device)
        saturation_attr = saturation_attr.to(device)
        hue_attr = hue_attr.to(device)
        blur_attr = blur_attr.to(device)

        overall_text = _fix_text_feature_dim(overall_text, device)
        brightness_text = _fix_text_feature_dim(brightness_text, device)
        contrast_text = _fix_text_feature_dim(contrast_text, device)
        saturation_text = _fix_text_feature_dim(saturation_text, device)
        hue_text = _fix_text_feature_dim(hue_text, device)
        blur_text = _fix_text_feature_dim(blur_text, device)

        y_pred = model(
            x, brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr,
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
            loss = (weights * sample_losses).mean()

            error_tracker.update(image_ids, sample_errors.detach())
            error_weights.update(weights.mean().item(), x.size(0))
        else:
            loss = criterion(y_pred, y)
            if ERROR_AWARE_ENABLED and error_tracker is not None:
                sample_errors = torch.abs(y_pred - y).view(-1).detach()
                error_tracker.update(image_ids, sample_errors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.update(loss.item(), x.size(0))

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
    true_scores = []
    pred_scores = []

    with torch.no_grad():
        for batch in tqdm(loader):
            (x, y,
             brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr,
             overall_text, brightness_text, contrast_text, saturation_text, hue_text, blur_text, image_ids) = batch

            x = x.to(device)
            y = y.type(torch.FloatTensor).to(device).view(y.size(0), -1)
            brightness_attr = brightness_attr.to(device)
            contrast_attr = contrast_attr.to(device)
            saturation_attr = saturation_attr.to(device)
            hue_attr = hue_attr.to(device)
            blur_attr = blur_attr.to(device)

            overall_text = _fix_text_feature_dim(overall_text, device)
            brightness_text = _fix_text_feature_dim(brightness_text, device)
            contrast_text = _fix_text_feature_dim(contrast_text, device)
            saturation_text = _fix_text_feature_dim(saturation_text, device)
            hue_text = _fix_text_feature_dim(hue_text, device)
            blur_text = _fix_text_feature_dim(blur_text, device)

            y_pred = model(
                x, brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr,
                overall_text, brightness_text, contrast_text, saturation_text, hue_text, blur_text,
                image_ids=image_ids
            )

            pscore_np = y_pred.data.cpu().numpy().astype('float')
            tscore_np = y.data.cpu().numpy().astype('float')
            pred_scores += pscore_np.mean(axis=1).tolist()
            true_scores += tscore_np.mean(axis=1).tolist()

            loss = criterion(y_pred, y)
            validate_losses.update(loss.item(), x.size(0))

    srcc_mean, _ = spearmanr(pred_scores, true_scores)
    lcc_mean, _ = pearsonr(pred_scores, true_scores)
    mae = mean_absolute_error(true_scores, pred_scores)
    mse = mean_squared_error(true_scores, pred_scores)
    true_scores = np.array(true_scores)
    pred_scores = np.array(pred_scores)
    true_label = np.where(true_scores <= 0.55, 0, 1)
    pred_label = np.where(pred_scores <= 0.55, 0, 1)
    acc = accuracy_score(true_label, pred_label)

    print('accuracy: {:.4f}, lcc_mean: {:.4f}, srcc_mean: {:.4f}, mae: {:.4f}, mse: {:.4f}, validate_losses: {:.4f}'.format(
        acc, lcc_mean, srcc_mean, mae, mse, validate_losses.avg))
    return validate_losses.avg, acc, lcc_mean, srcc_mean


def start_train(opt):
    device = torch.device(f"cuda:{opt['gpu_id']}")
    train_loader, val_loader = create_data_part(opt)

    model = HKDFusion(
        pretrained_cfg_path=opt.get('swin_weight_path'),
        target_feature_size=14,
        visual_dim=256,
        saga_version=SAGA_VERSION,
        sensitivity_csv_path=SENSITIVITY_CSV_PATH
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt['init_lr'])
    criterion = torch.nn.MSELoss().to(device)

    resume_path = opt.get('resume', '')
    start_epoch = 0
    srcc_best = 0.0
    vacc_best = 0.0

    if resume_path and os.path.exists(resume_path):
        print(f"🔄 从checkpoint恢复训练: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0)
        srcc_best = checkpoint.get('srcc_val', 0.0)
        vacc_best = checkpoint.get('vacc', 0.0)
        print(f"📅 从第 {start_epoch} 个epoch继续训练，SRCC_best={srcc_best:.4f}, ACC_best={vacc_best:.4f}")
    elif resume_path:
        print(f"⚠️ 指定的checkpoint不存在: {resume_path}，从头开始训练")

    save_dir = CONFIG_CKPT_DIR or opt["path_to_save_ckpt"]
    for e in range(start_epoch, opt['num_epoch']):
        adjust_learning_rate(opt, optimizer, e)
        train_loss = train(opt, model, train_loader, optimizer, criterion, device, epoch=e)
        val_loss, vacc, vlcc, vsrcc = validate(opt, model, val_loader, criterion, device)
        print(f"Epoch {e+1}/{opt['num_epoch']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'epoch_{e+1:03d}.pth')
        torch.save({
            'epoch': e+1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'srcc_val': vsrcc,
            'plcc_val': vlcc,
            'vacc': vacc
        }, save_path)
        print(f"✅ Saved model to {save_path}")

        srcc_best = max(srcc_best, vsrcc)
        vacc_best = max(vacc_best, vacc)

        if NNI_AVAILABLE:
            nni.report_intermediate_result({'default': vacc, "vsrcc": vsrcc, "val_loss": val_loss})

    if NNI_AVAILABLE:
        nni.report_final_result({'default': vacc_best, "vsrcc": srcc_best})


if __name__ == "__main__":
    base_opt = option.init()
    if NNI_AVAILABLE:
        tuner_params = nni.get_next_parameter()
        merged_opt = vars(merge_parameter(base_opt, tuner_params))
    else:
        merged_opt = vars(base_opt)
    start_train(merged_opt)

