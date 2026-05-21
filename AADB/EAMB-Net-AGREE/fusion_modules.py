"""
AGREE (Attribute-aware Gradient conflict REduction for aEsthetics) Fusion Modules

论文核心组件实现：
1. **属性解耦表示** (Attribute-aware Decoupling)
   - 显式建模6个属性维度（overall + 5 specific: Brightness, Contrast, Blur, Hue, Saturation）
   - 每个属性都有独立的视觉特征和文本特征
   - 对应论文公式(1)(2)

2. **多模态属性融合** (Multimodal Fusion)
   - 对每个属性进行视觉-文本融合
   - 使用LLaVA提取的文本特征作为语义锚点
   - 对应论文公式(3)(4)

3. **敏感度引导加权** (Sensitivity-guided Routing)
   - 根据属性敏感度动态分配权重
   - 对应论文公式(5)(6)(7)(8)

本文件实现论文中使用的 SensitivityGuidedFusion 模块。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os


class SimpleVisualTextFusion(nn.Module):
    """
    视觉-文本简单融合模块
    
    对应论文公式(4): h_k = ReLU(Conv_{1x1}^{2C->C}([h_k^v; H_k^t]))
    
    特点：
    - 极简设计：只用1x1卷积
    - 无BatchNorm：避免小数据集过拟合
    """
    
    def __init__(self, visual_dim=256, text_dim=4096, spatial_size=64):
        super(SimpleVisualTextFusion, self).__init__()
        
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.spatial_size = spatial_size
        
        # 文本特征降维（简单线性层）
        self.text_adapter = nn.Linear(text_dim, visual_dim)
        
        # 简单融合：拼接后1x1卷积（无BN，避免过拟合）
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(visual_dim * 2, visual_dim, 1),  # 1x1卷积
            nn.ReLU()
        )
        
        print(f"✅ AGREE SimpleVisualTextFusion 初始化完成")
    
    def forward(self, visual_feature, text_feature):
        """
        Args:
            visual_feature: [B, 256, H, W] 视觉特征
            text_feature: [B, 4096] 文本特征
            
        Returns:
            fused_feature: [B, 256, H, W] 融合后的特征
        """
        B, C, H, W = visual_feature.shape
        
        # 1. 文本特征降维
        text_adapted = self.text_adapter(text_feature)  # [B, 256]
        
        # 2. 扩展到空间维度
        text_spatial = text_adapted.unsqueeze(-1).unsqueeze(-1).expand(B, C, H, W)
        
        # 3. 简单拼接和融合
        concat_features = torch.cat([visual_feature, text_spatial], dim=1)  # [B, 512, H, W]
        fused_feature = self.fusion_conv(concat_features)  # [B, 256, H, W]
        
        return fused_feature


# =============================================================================
# 以下为消融实验用的备选模块（论文未使用，已注释）
# =============================================================================

# class SimpleConcatFusion(nn.Module):
#     """
#     简单拼接融合模块（无敏感度引导）- 消融实验用
#     """
#     
#     def __init__(self, feature_dim=256, num_modalities=6, output_dim=256):
#         super(SimpleConcatFusion, self).__init__()
#         
#         self.feature_dim = feature_dim
#         self.num_modalities = num_modalities
#         self.output_dim = output_dim
#         
#         self.fusion_conv = nn.Sequential(
#             nn.Conv2d(feature_dim * num_modalities, output_dim, 1),
#             nn.ReLU()
#         )
#     
#     def forward(self, overall_feature, attr_features, image_ids=None):
#         all_features = [overall_feature] + attr_features
#         concat_features = torch.cat(all_features, dim=1)
#         fused_feature = self.fusion_conv(concat_features)
#         return fused_feature


# =============================================================================
# 论文使用的核心模块
# =============================================================================

class SensitivityGuidedFusion(nn.Module):
    """
    AGREE 敏感度引导融合模块（论文核心组件）
    
    对应论文 Section 4.3 Sensitivity-guided Routing:
    - 公式(5): s_k(I) = |f(I^{+k}) - f(I^{-k})| / 2Δ  (敏感度计算，离线预计算)
    - 公式(6): α_k(I) = s_k(I) / Σs_j(I)  (归一化权重)
    - 公式(7): F_attr = Conv_{1x1}^{5C->C}(Concat(α_1*h_1, ..., α_5*h_5))
    - 公式(8): F_final = Conv_{1x1}^{2C->C}(Concat(h_0, F_attr))
    """
    
    def __init__(self, feature_dim=256, output_dim=256, sensitivity_csv_path=None):
        super(SensitivityGuidedFusion, self).__init__()
        
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.sensitivity_csv_path = sensitivity_csv_path
        
        # 加载敏感度数据
        self.sensitivity_data = None
        if sensitivity_csv_path and os.path.exists(sensitivity_csv_path):
            self.sensitivity_data = pd.read_csv(sensitivity_csv_path)
            self.sensitivity_data.set_index('image_id', inplace=True)
            print(f"✅ 加载敏感度数据: {len(self.sensitivity_data)} 条记录")
        else:
            print("⚠️  未提供敏感度CSV文件，将使用均匀权重（退化为SimpleConcatFusion）")
        
        # 属性名称映射（与CSV列顺序一致）
        self.attr_names = ['brightness', 'contrast', 'blur', 'hue', 'saturation']
        
        # 属性融合网络（简单设计：1x1卷积，避免过拟合）
        self.attr_fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 5, feature_dim, 1),  # 5个属性 → 1个融合特征
            nn.ReLU()
        )
        
        # 最终融合：overall + 融合后的属性
        self.final_fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 2, output_dim, 1),  # overall + attr → output
            nn.ReLU()
        )
        
        print(f"✅ AGREE SensitivityGuidedFusion 初始化完成")
    
    def get_sensitivity_weights(self, image_ids):
        """
        根据图像ID从CSV获取敏感度权重
        
        Returns:
            weights: [B, 5] 归一化的敏感度权重
        """
        batch_size = len(image_ids)
        weights = torch.zeros(batch_size, 5)
        
        if self.sensitivity_data is None:
            # 如果没有敏感度数据，使用均匀权重（退化为简单拼接）
            weights = torch.ones(batch_size, 5) / 5.0
        else:
            for i, img_id in enumerate(image_ids):
                try:
                    # 从CSV中读取敏感度值（顺序: brightness, contrast, blur, hue, saturation）
                    row = self.sensitivity_data.loc[img_id]
                    sens_values = [
                        row['sens_brightness'],
                        row['sens_contrast'],
                        row['sens_blur'],
                        row['sens_hue'],
                        row['sens_saturation']
                    ]
                    # 归一化为权重
                    sens_tensor = torch.tensor(sens_values, dtype=torch.float32)
                    sens_tensor = torch.clamp(sens_tensor, min=0.0)  # 确保非负
                    weights[i] = sens_tensor / (sens_tensor.sum() + 1e-8)
                except KeyError:
                    # 如果找不到该图像，使用均匀权重
                    weights[i] = torch.ones(5) / 5.0
        
        return weights
    
    def forward(self, overall_feature, attr_features, image_ids=None):
        """
        前向传播
        
        Args:
            overall_feature: [B, C, H, W] overall特征 (h_0)
            attr_features: list of [B, C, H, W]，包含5个属性特征
                          顺序: [brightness, contrast, blur, hue, saturation]
            image_ids: list of str, 图像文件名（用于查询敏感度）
            
        Returns:
            fused_feature: [B, C, H, W] 最终融合特征 (F_final)
        """
        B, C, H, W = overall_feature.shape
        device = overall_feature.device
        
        # Step 1: 获取敏感度权重 α_k(I) - 公式(6)
        if image_ids is not None:
            sensitivity_weights = self.get_sensitivity_weights(image_ids).to(device)  # [B, 5]
        else:
            sensitivity_weights = torch.ones(B, 5, device=device) / 5.0
        
        # Step 2: 根据敏感度加权属性特征 α_k * h_k
        weighted_attr_features = []
        for i, attr_feat in enumerate(attr_features):
            weight = sensitivity_weights[:, i:i+1, None, None]  # [B, 1, 1, 1]
            weighted_feat = attr_feat * weight  # [B, C, H, W]
            weighted_attr_features.append(weighted_feat)
        
        # Step 3: 拼接加权属性特征 - 公式(7)
        concat_attr = torch.cat(weighted_attr_features, dim=1)  # [B, 5C, H, W]
        fused_attr = self.attr_fusion(concat_attr)  # [B, C, H, W] = F_attr
        
        # Step 4: overall与属性融合 - 公式(8)
        final_concat = torch.cat([overall_feature, fused_attr], dim=1)  # [B, 2C, H, W]
        fused_feature = self.final_fusion(final_concat)  # [B, C, H, W] = F_final
        
        return fused_feature


# =============================================================================
# 以下为消融实验用的多任务学习模块（论文未使用，已注释）
# =============================================================================

# class SensitivityGuidedFusionWithMTL(nn.Module):
#     """
#     敏感度引导融合 + 多任务学习 - 消融实验用
#     """
#     
#     def __init__(self, feature_dim=256, output_dim=256, sensitivity_csv_path=None):
#         super(SensitivityGuidedFusionWithMTL, self).__init__()
#         
#         self.base_fusion = SensitivityGuidedFusion(
#             feature_dim=feature_dim,
#             output_dim=output_dim,
#             sensitivity_csv_path=sensitivity_csv_path
#         )
#         
#         self.sensitivity_predictor = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Linear(output_dim, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 5),
#             nn.Softmax(dim=1)
#         )
#     
#     def forward(self, overall_feature, attr_features, image_ids=None, return_sensitivity=False):
#         fused_feature = self.base_fusion(overall_feature, attr_features, image_ids)
#         
#         if return_sensitivity:
#             pred_sensitivity = self.sensitivity_predictor(fused_feature)
#             return fused_feature, pred_sensitivity
#         else:
#             return fused_feature


# =============================================================================
# 模块获取接口
# =============================================================================

def get_fusion_module(version='v2', feature_dim=256, output_dim=256, sensitivity_csv_path=None):
    """
    获取AGREE融合模块
    
    Args:
        version: 'v2' (敏感度引导，论文使用)
        feature_dim: 特征维度
        output_dim: 输出维度
        sensitivity_csv_path: 敏感度CSV文件路径
        
    Returns:
        fusion_module: SensitivityGuidedFusion 模块
    """
    if version == 'v2':
        print("📦 AGREE: 使用 SensitivityGuidedFusion")
        return SensitivityGuidedFusion(feature_dim, output_dim, sensitivity_csv_path)
    else:
        raise ValueError(f"AGREE 论文使用 version='v2'，当前输入: {version}")


# =============================================================================
# 测试函数
# =============================================================================

def test_fusion_modules():
    """测试AGREE融合模块"""
    print("="*80)
    print("🧪 测试 AGREE 融合模块")
    print("="*80)
    
    # 创建测试数据
    B, C, H, W = 4, 256, 64, 64
    overall_feature = torch.randn(B, C, H, W)
    attr_features = [torch.randn(B, C, H, W) for _ in range(5)]
    image_ids = [f"test_{i}.jpg" for i in range(B)]
    
    print("\n测试 SensitivityGuidedFusion (论文使用)")
    print("-"*40)
    fusion = get_fusion_module('v2', feature_dim=256, output_dim=256, sensitivity_csv_path=None)
    output = fusion(overall_feature, attr_features, image_ids)
    print(f"输入: overall {overall_feature.shape}, 5个属性特征")
    print(f"输出: {output.shape}")
    
    print("\n" + "="*80)
    print("✅ AGREE 融合模块测试通过！")
    print("="*80)


if __name__ == "__main__":
    test_fusion_modules()