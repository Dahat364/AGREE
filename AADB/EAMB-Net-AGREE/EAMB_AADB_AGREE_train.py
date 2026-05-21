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
from multimodal_dataset import MultimodalAADBDataset
from util import AverageMeter


def load_config(config_path='config.yml'):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# 全局配置
AGREE_CONFIG = load_config()
SENSITIVITY_CSV_PATH = AGREE_CONFIG.get('sensitivity_csv_path', None)
CONFIG_CKPT_DIR = AGREE_CONFIG.get('checkpoint_dir', None)

# AGREE：误差感知加权配置 - 对应论文公式(10)(11)(12)
ERROR_AWARE_CONFIG = AGREE_CONFIG.get('error_aware', {})
ERROR_AWARE_ENABLED = ERROR_AWARE_CONFIG.get('enabled', False)
ERROR_AWARE_BETA = ERROR_AWARE_CONFIG.get('beta', 1.0)
ERROR_AWARE_TAU = ERROR_AWARE_CONFIG.get('tau', 0.1)
ERROR_AWARE_WARMUP = ERROR_AWARE_CONFIG.get('warmup_epochs', 5)
ERROR_AWARE_EMA_DECAY = ERROR_AWARE_CONFIG.get('ema_decay', 0.9)

print(f"\n{'='*80}")
print(f"🚀 EAMB-Net + AGREE 训练脚本 (AADB)")
print(f"   敏感度CSV: {SENSITIVITY_CSV_PATH}")
if CONFIG_CKPT_DIR:
    print(f"   checkpoint_dir: {CONFIG_CKPT_DIR}")
if ERROR_AWARE_ENABLED:
    print(f"   误差感知加权(EA-MSE): ✅ 启用 (β={ERROR_AWARE_BETA}, τ={ERROR_AWARE_TAU}, warmup={ERROR_AWARE_WARMUP})")
else:
    print(f"   误差感知加权(EA-MSE): ❌ 未启用")
print(f"{'='*80}\n")


class ErrorTracker:
    """
    AGREE 误差感知加权 (Error-Aware Reweighting)
    
    对应论文 Section 4.4:
    - 公式(10): e_i^{(t)} = γ * e_i^{(t-1)} + (1-γ) * |y_i - ŷ_i^{(t)}|, γ=0.9
    - 公式(11): ω_i = 1 + β * tanh(max(0, (e_i - μ_g) / (τ * σ_g))), β=1.0, τ=0.1
    - 公式(12): L = (1/N) * Σ ω̃_i * (ŷ_i - y_i)²
    """
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


error_tracker = ErrorTracker(ema_decay=ERROR_AWARE_EMA_DECAY) if ERROR_AWARE_ENABLED else None


# 注：敏感度数据由 SensitivityGuidedFusion 模块内部加载，无需在训练脚本中处理


def create_data_part(opt):
    """创建数据加载器"""
    csv_root = opt['path_to_save_csv']
    images_path = opt['path_to_images']
    text_root = opt['path_to_text_features']

    train_ds = MultimodalAADBDataset(os.path.join(csv_root, 'train.csv'), images_path, text_root, if_train=True)
    val_ds = MultimodalAADBDataset(os.path.join(csv_root, 'test.csv'), images_path, text_root, if_train=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    return train_loader, val_loader


def _freeze_emotion_model(model):
    """冻结情感模型"""
    for param in model.emotion_model.parameters():
        param.requires_grad = False


def _prepare_batch(batch, device):
    """准备批次数据"""
    (x, y,
     brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr,
     overall_text, brightness_text, contrast_text, saturation_text, hue_text, blur_text,
     image_ids) = batch

    tensors = [
        brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr
    ]
    text_tensors = [
        overall_text, brightness_text, contrast_text, saturation_text, hue_text, blur_text
    ]

    x = x.to(device)
    y = y.to(device).view(y.size(0), -1).float()
    attr_tensors = [t.to(device) for t in tensors]
    text_tensors = [t.to(device).float() for t in text_tensors]

    return x, y, attr_tensors, text_tensors, image_ids


def train(opt, model, loader, optimizer, criterion, device, epoch=0):
    """
    训练一个epoch
    
    AGREE组件:
    - Error-Aware Reweighting: 公式(10)(11)(12)
    """
    model.train()
    _freeze_emotion_model(model)
    train_losses = AverageMeter()
    error_weights = AverageMeter()

    use_error_aware = (
        ERROR_AWARE_ENABLED and
        epoch >= ERROR_AWARE_WARMUP and
        error_tracker is not None
    )
    if use_error_aware:
        print(f"   🎯 AGREE EA-MSE 已启用 (epoch {epoch+1} > warmup {ERROR_AWARE_WARMUP})")

    for batch in tqdm(loader):
        x, y, attr_tensors, text_tensors, image_ids = _prepare_batch(batch, device)
        (brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr) = attr_tensors
        (overall_text, brightness_text, contrast_text, saturation_text, hue_text, blur_text) = text_tensors

        # 前向传播
        y_pred = model(
            x,
            brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr,
            overall_text, brightness_text, contrast_text, saturation_text, hue_text, blur_text,
            image_ids=image_ids
        )
        
        # 计算损失 - 公式(12)
        if use_error_aware:
            sample_errors = torch.abs(y_pred - y).view(-1)
            weights = error_tracker.get_weights(
                image_ids, sample_errors.detach(),
                beta=ERROR_AWARE_BETA,
                tau=ERROR_AWARE_TAU
            ).to(device)
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

    # 打印训练信息
    if ERROR_AWARE_ENABLED and error_tracker is not None:
        _, _, n_samples = error_tracker.get_stats()
        if use_error_aware:
            print(f"   Train - Loss: {train_losses.avg:.6f}, AvgWeight: {error_weights.avg:.3f}")
        else:
            print(f"   Train - Loss: {train_losses.avg:.6f} (预热中, epoch {epoch+1}/{ERROR_AWARE_WARMUP})")
        print(f"   Error Stats - μ_g: {error_tracker.global_mean:.4f}, σ_g: {error_tracker.global_std:.4f}, Tracked: {n_samples}")
    else:
        print(f"   Train - Loss: {train_losses.avg:.6f}")
    
    return train_losses.avg


def validate(opt, model, loader, criterion, device):
    """验证模型"""
    model.eval()
    validate_losses = AverageMeter()
    true_scores, pred_scores = [], []

    with torch.no_grad():
        for batch in tqdm(loader):
            x, y, attr_tensors, text_tensors, image_ids = _prepare_batch(batch, device)
            (brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr) = attr_tensors
            (overall_text, brightness_text, contrast_text, saturation_text, hue_text, blur_text) = text_tensors

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
    true_label = np.where(true_scores <= 0.5, 0, 1)
    pred_label = np.where(pred_scores <= 0.5, 0, 1)
    acc = accuracy_score(true_label, pred_label)

    print('accuracy: {:.4f}, lcc_mean: {:.4f}, srcc_mean: {:.4f}, mae: {:.4f}, mse: {:.4f}, rmse: {:.4f}'.format(
        acc, lcc_mean, srcc_mean, mae, mse, rmse))

    return validate_losses.avg, acc, srcc_mean, lcc_mean


def start_train(opt):
    """开始训练"""
    device = torch.device(f"cuda:{opt['gpu_id']}")
    train_loader, val_loader = create_data_part(opt)

    model = EAMBNet_AGREE(
        sensitivity_csv_path=SENSITIVITY_CSV_PATH
    ).to(device)
    
    criterion = nn.MSELoss().to(device)

    # 分层学习率设置
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

    save_dir = CONFIG_CKPT_DIR or opt["path_to_save_ckpt"]
    for epoch in range(start_epoch, opt['num_epoch']):
        print(f"\n===== Epoch {epoch + 1}/{opt['num_epoch']} =====")
        train_loss = train(opt, model, train_loader, optimizer, criterion, device, epoch=epoch)
        val_loss, vacc, vsrcc, vlcc = validate(opt, model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{opt['num_epoch']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

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
            nni.report_intermediate_result(
                {'default': vacc, "vsrcc": vsrcc, "val_loss": val_loss})

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
