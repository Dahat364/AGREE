import os
import warnings
import torch
import numpy as np
import yaml
from torch import nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr, pearsonr

import option
from models.EAMBNet_AGREE import EAMBNet_AGREE
from multimodal_dataset import MultimodalAVADataset


def load_config(config_path='config.yml'):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# 全局配置
SAGA_CONFIG = load_config()
SAGA_VERSION = SAGA_CONFIG.get('saga_version', 'v2')
SENSITIVITY_CSV_PATH = SAGA_CONFIG.get('sensitivity_csv_path', None)
CONFIG_CKPT_DIR = SAGA_CONFIG.get('checkpoint_dir', None)

print(f"\n{'='*80}")
print(f"🧪 EAMB-Net AGREE测试脚本（读取config.yml）")
if SAGA_CONFIG.get('version'):
    print(f"   config.version: {SAGA_CONFIG.get('version')}")
print(f"   敏感度CSV: {SENSITIVITY_CSV_PATH if SENSITIVITY_CSV_PATH else '无（v1模式）'}")
if CONFIG_CKPT_DIR:
    print(f"   checkpoint_dir: {CONFIG_CKPT_DIR}")
print(f"{'='*80}\n")


def _prepare_batch(batch, device):
    """准备批次数据"""
    (x, y,
     brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr,
     overall_text, brightness_text, contrast_text, saturation_text, hue_text, blur_text,
     image_ids) = batch

    x = x.to(device)
    y = y.to(device).view(y.size(0), -1).float()
    attr_tensors = [brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr]
    attr_tensors = [t.to(device) for t in attr_tensors]
    text_tensors = [overall_text, brightness_text, contrast_text, saturation_text, hue_text, blur_text]
    text_tensors = [t.to(device).float() for t in text_tensors]
    return x, y, attr_tensors, text_tensors, image_ids


def get_ava_score(y_pred, device):
    """计算AVA评分（加权求和 1-10）"""
    # ⚠️ 重要：归一化y_pred，确保它是概率分布（和为1）
    y_pred_norm = y_pred / y_pred.sum(dim=1, keepdim=True)
    w = torch.from_numpy(np.linspace(1, 10, 10)).type(torch.FloatTensor).to(device)
    w_batch = w.repeat(y_pred.size(0), 1)
    score = (y_pred_norm * w_batch).sum(dim=1)
    return score

def create_test_loader(opt):
    """创建测试数据加载器"""
    test_csv_path = os.path.join(opt['path_to_save_csv'], 'test.csv')
    dataset = MultimodalAVADataset(
        test_csv_path,
        opt['path_to_images'],
        opt['path_to_text_features'],
        if_train=False
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opt['batch_size'],
        num_workers=opt['num_workers'],
        shuffle=False
    )


def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()
    pred_scores, true_scores, losses = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader):
            x, y, attr_tensors, text_tensors, image_ids = _prepare_batch(batch, device)
            (brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr) = attr_tensors
            (overall_text, brightness_text, contrast_text, saturation_text, hue_text, blur_text) = text_tensors

            # 根据版本调用模型
            if SAGA_VERSION == 'v3':
                # v3版本返回两个输出
                y_pred, pred_sensitivity = model(
                    x,
                    brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr,
                    overall_text, brightness_text, contrast_text, saturation_text, hue_text, blur_text,
                    image_ids=image_ids,
                    return_sensitivity=True
                )
            else:
                # v1和v2版本只返回美学分数
                y_pred = model(
                    x,
                    brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr,
                    overall_text, brightness_text, contrast_text, saturation_text, hue_text, blur_text,
                    image_ids=image_ids
                )

            # 测试时只使用美学分数
            loss = criterion(y_pred, y)
            losses.append(loss.item())

            # AVA评分计算：加权求和 (1-10)
            pred_score_values = get_ava_score(y_pred, device)
            true_score_values = get_ava_score(y, device)
            pred_scores += pred_score_values.cpu().numpy().tolist()
            true_scores += true_score_values.cpu().numpy().tolist()

    pred_scores = np.array(pred_scores)
    true_scores = np.array(true_scores)

    srcc, _ = spearmanr(pred_scores, true_scores)
    plcc, _ = pearsonr(pred_scores, true_scores)
    mae = mean_absolute_error(true_scores, pred_scores)
    mse = mean_squared_error(true_scores, pred_scores)
    rmse = np.sqrt(mse)
    # AVA准确率阈值：5.5分
    acc = accuracy_score(
        (true_scores > 5.5).astype(int),
        (pred_scores > 5.5).astype(int)
    )

    print("\n🎯 [Test Results]")
    print(f"SRCC:  {srcc:.4f}")
    print(f"PLCC:  {plcc:.4f}")
    print(f"MAE:   {mae:.4f}")
    print(f"MSE:   {mse:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"ACC:   {acc:.4f}")
    print(f"Loss:  {np.mean(losses):.4f}")
    return srcc, plcc, mae, mse, rmse, acc


def _resolve_checkpoint(opt, explicit_path=None):
    """解析checkpoint路径"""
    if explicit_path:
        if os.path.exists(explicit_path):
            return explicit_path
        raise FileNotFoundError(f"指定的权重文件不存在: {explicit_path}")
    ckpt_dir = CONFIG_CKPT_DIR or opt.get('path_to_save_ckpt')
    if not ckpt_dir:
        raise KeyError("未提供 checkpoint_dir（config.yml）或 path_to_save_ckpt（option.py）")
    ckpts = sorted(
        [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
    )
    if not ckpts:
        raise FileNotFoundError(f"未在 {ckpt_dir} 找到任何 checkpoint")
    return ckpts[-1]


def run_test(opt, ckpt_path=None):
    """运行测试"""
    device = torch.device(f"cuda:{opt['gpu_id']}")
    
    model = EAMBNet_AGREE(
        saga_version=SAGA_VERSION,
        sensitivity_csv_path=SENSITIVITY_CSV_PATH,
        dataset='AVA'  # 指定AVA数据集
    ).to(device)
    
    ckpt_path = _resolve_checkpoint(opt, ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    print(f"✅ Loaded model from {ckpt_path}")

    loader = create_test_loader(opt)
    criterion = nn.MSELoss().to(device)
    evaluate(model, loader, criterion, device)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    opt = vars(option.init())
    
    # 单个权重测试（取消注释使用）
    # weight_path = "/path/to/checkpoint/epoch_050.pth"
    # run_test(opt, weight_path)
    
    # 批量测试（根据需要修改路径）
    checkpoint_dir = CONFIG_CKPT_DIR or opt.get('path_to_save_ckpt', 'checkpoints')
    print(f"📁 扫描checkpoint目录: {checkpoint_dir}")
    
    if os.path.exists(checkpoint_dir):
        ckpts = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
        if ckpts:
            print(f"找到 {len(ckpts)} 个checkpoints")
            for ckpt_file in ckpts:
                weight_path = os.path.join(checkpoint_dir, ckpt_file)
                print(f"\n{'='*80}")
                print(f"测试: {ckpt_file}")
                print(f"{'='*80}")
                run_test(opt, weight_path)
        else:
            print("⚠️  未找到任何checkpoint文件")
    else:
        print(f"⚠️  Checkpoint目录不存在: {checkpoint_dir}")
