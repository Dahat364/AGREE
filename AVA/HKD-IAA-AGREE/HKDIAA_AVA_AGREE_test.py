import os
import warnings
from tqdm import tqdm
import torch
import numpy as np
import yaml
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

import option
from models.HKD_Fusion import HKDFusion
from multimodal_dataset import MultimodalAVADataset

warnings.filterwarnings('ignore')


def load_config(config_path='config.yml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


SAGA_CONFIG = load_config()
SAGA_VERSION = SAGA_CONFIG.get('saga_version', 'v2')
SENSITIVITY_CSV_PATH = SAGA_CONFIG.get('sensitivity_csv_path', None)
CONFIG_CKPT_DIR = SAGA_CONFIG.get('checkpoint_dir', None)

print(f"\n{'='*80}")
print(f"🧪 HKD-IAA AGREE测试脚本（读取config.yml）")
if SAGA_CONFIG.get('version'):
    print(f"   config.version: {SAGA_CONFIG.get('version')}")
print(f"   敏感度CSV: {SENSITIVITY_CSV_PATH if SENSITIVITY_CSV_PATH else '无（v1模式）'}")
if CONFIG_CKPT_DIR:
    print(f"   checkpoint_dir: {CONFIG_CKPT_DIR}")
print(f"{'='*80}\n")


def _fix_text_feature_dim(text_feat, device):
    text_feat = text_feat.to(device).float()
    if text_feat.dim() == 3:
        text_feat = text_feat.squeeze(1)
    elif text_feat.dim() == 1:
        text_feat = text_feat.unsqueeze(0)
    return text_feat


def validate(opt, model, loader, criterion, device):
    model.eval()
    true_scores = []
    pred_scores = []
    
    # ✅ AVA评分权重：[1, 2, 3, ..., 10] 对应10个分数档次
    w = torch.from_numpy(np.linspace(1, 10, 10)).type(torch.FloatTensor).to(device)

    with torch.no_grad():
        for batch in tqdm(loader):
            (x, y,
             brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr,
             overall_text, brightness_text, contrast_text, saturation_text, hue_text, blur_text, image_id_clean) = batch

            x = x.to(device)
            y = y.type(torch.FloatTensor).to(device)
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
                image_ids=image_id_clean
            )

            # ✅ 修复：使用加权求和计算AVA分数，而不是简单平均（与训练保持一致）
            # ⚠️ 重要：归一化y_pred和y，确保它们是概率分布（和为1）
            y_pred_norm = y_pred / y_pred.sum(dim=1, keepdim=True)
            y_norm = y / y.sum(dim=1, keepdim=True)
            
            w_batch = w.repeat(y_pred.size(0), 1)
            pred_scores += (y_pred_norm * w_batch).sum(dim=1).cpu().numpy().tolist()
            true_scores += (y_norm * w_batch).sum(dim=1).cpu().numpy().tolist()

            loss = criterion(y, y_pred)

    srcc_mean, _ = spearmanr(pred_scores, true_scores)
    plcc_mean, _ = pearsonr(pred_scores, true_scores)
    mae = mean_absolute_error(true_scores, pred_scores)
    mse = mean_squared_error(true_scores, pred_scores)
    rmse = np.sqrt(mse)
    true_scores = np.array(true_scores)
    pred_scores = np.array(pred_scores)
    # ✅ 修复：分数范围是1-10，阈值应该是5.5而不是0.55（与训练保持一致）
    true_label = np.where(true_scores <= 5.5, 0, 1)
    pred_label = np.where(pred_scores <= 5.5, 0, 1)
    acc = accuracy_score(true_label, pred_label)

    print("\n🎯 [Test Results]")
    print(f"SRCC:  {srcc_mean:.4f}")
    print(f"PLCC:  {plcc_mean:.4f}")
    print(f"MAE:   {mae:.4f}")
    print(f"MSE:   {mse:.4f}")
    print(f"RMSE:   {rmse:.4f}")
    print(f"ACC:   {acc:.4f}")
    return plcc_mean, srcc_mean, mae, mse, acc, rmse


def load_test_loader(opt):
    csv_root = opt.path_to_PARA_save_csv
    images_path = opt.path_to_PARA_images
    text_root = opt.path_to_text_features

    test_dataset = MultimodalAVADataset(
        os.path.join(csv_root, 'test.csv'),
        images_path,
        text_root,
        if_train=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.test_num_workers,
        shuffle=False
    )
    return test_loader


def load_model_and_test(opt, ckpt_path):
    device = torch.device(f"cuda:{opt.gpu_id}")
    print(f"✅ Loading model from {ckpt_path}")

    model = HKDFusion(
        dataset='AVA',  # ⭐ 指定AVA数据集
        pretrained_cfg_path=opt.swin_weight_path,
        target_feature_size=14,
        visual_dim=256,
        saga_version=SAGA_VERSION,
        sensitivity_csv_path=SENSITIVITY_CSV_PATH
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    criterion = torch.nn.MSELoss().to(device)
    test_loader = load_test_loader(opt)
    validate(opt, model, test_loader, criterion, device)


if __name__ == "__main__":
    opt = option.init()
    warnings.filterwarnings('ignore')

    # 单个权重测试（取消注释使用）
    # ckpt_path = '/home/dmc/12TB_ZHZ68JLF/A_PROJECT_IAA/CheckPoints/MyWork/AVA/HKD-IAA-AGREE/epoch_014.pth'
    # assert os.path.exists(ckpt_path), f"❌ 模型路径不存在: {ckpt_path}"
    # load_model_and_test(opt, ckpt_path)

    # 批量测试（根据需要修改路径）
    checkpoint_dir = CONFIG_CKPT_DIR or opt.path_to_save_ckpt
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
                load_model_and_test(opt, weight_path)
        else:
            print("⚠️  未找到任何checkpoint文件")
    else:
        print(f"⚠️  Checkpoint目录不存在: {checkpoint_dir}")
