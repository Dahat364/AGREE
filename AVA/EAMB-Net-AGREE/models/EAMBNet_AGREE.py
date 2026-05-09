"""
EAMBNet with SAGA (Sensitivity-Aware Guidance for Aesthetics)

基于EAMBNet架构，集成SAGA多模态融合模块
支持v1/v2/v3三个版本的消融实验
"""

import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.EmotionNet import EmoClassifier
from fusion_modules import SimpleVisualTextFusion, get_fusion_module


def emotion_model():
    """加载预训练的情感模型"""
    emotion_model = EmoClassifier()
    # 使用dmc的路径
    emotion_model_path = '/home/dmc/12TB_ZHZ68JLF/A_PROJECT_IAA/weights/EMBA-NET/emotion_model.pth'
    if not os.path.exists(emotion_model_path):
        # 备用路径
        emotion_model_path = '/home/dmc/12TB_ZHZ68JLF/A_PROJECT_IAA/weights/EMBA-NET/emotion_model.pth'
    checkpoint_emotion_model = torch.load(emotion_model_path, map_location='cpu')
    emotion_model.load_state_dict(checkpoint_emotion_model['model'])
    return emotion_model


class CBAMLayer(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class SADEM(nn.Module):
    """Semantic-Aware Dual-path Enhancement Module"""
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        super(SADEM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU()
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU()
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels,
                          kernel_size=1, bias=False),
                BatchNorm(in_channels)
            )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))

        y = F.interpolate(y, size=[input_size[2], input_size[3]],
                          mode='bilinear', align_corners=False)

        x = sim_map * x + sim_map * y + x

        return x


class conv_bn_relu(nn.Module):
    """卷积-BN-ReLU模块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class up_conv_bn_relu(nn.Module):
    """上采样-卷积-BN-ReLU模块"""
    def __init__(self, up_size, in_channels, out_channels, kernal_size=1, padding=0, stride=1):
        super(up_conv_bn_relu, self).__init__()
        self.upSample = nn.Upsample(size=(up_size, up_size), mode="bilinear", align_corners=True)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernal_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.upSample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class EAMBNet_AGREE(nn.Module):
    """
    EAMBNet with multimodal fusion module
    
    Args:
        saga_version: 'v1', 'v2', or 'v3'
        sensitivity_csv_path: 敏感度CSV文件路径
        dataset: 数据集名称 ('AVA', 'TAD66K', 'LAPIS', 等)
    """
    def __init__(self, saga_version='v2', sensitivity_csv_path=None, dataset='TAD66K'):
        super(EAMBNet_AGREE, self).__init__()
        
        self.saga_version = saga_version
        self.sensitivity_csv_path = sensitivity_csv_path
        self.dataset = dataset
        
        print(f"\n{'='*80}")
        print(f"🚀 初始化 EAMBNet-AGREE ({saga_version}) for {dataset}")
        print(f"{'='*80}\n")
        
        # 情感模型
        self.emotion_model = emotion_model()
        for p in self.emotion_model.parameters():
            p.requires_grad = False
        
        # 加载ResNet50权重
        state_dict_path = "/home/dmc/12TB_ZHZ68JLF/A_PROJECT_IAA/weights/EMBA-NET/resnet50-0676ba61.pth"
        if not os.path.exists(state_dict_path):
            state_dict_path = "/home/dmc/12TB_ZHZ68JLF/A_PROJECT_IAA/weights/EMBA-NET/resnet50-0676ba61.pth"
        state_dict = torch.load(state_dict_path, map_location='cpu')
        
        # ============ 原图像的处理分支 ============
        self.model_x = torchvision.models.resnet50(pretrained=False)
        self.model_x.load_state_dict(state_dict)
        self.feature1_x = nn.Sequential(*list(self.model_x.children())[:5])
        self.feature2_x = list(self.model_x.children())[5]
        
        self.model_s = torchvision.models.resnet50(pretrained=False)
        self.model_s.load_state_dict(state_dict)
        self.feature1_s = nn.Sequential(*list(self.model_s.children())[:5])
        self.feature2_s = list(self.model_s.children())[5]
        self.feature3_s = list(self.model_s.children())[6]
        self.feature4_s = list(self.model_s.children())[7]

        # ============ 5个属性的共享处理分支 ============
        self.attr_model_x = torchvision.models.resnet50(pretrained=False)
        self.attr_model_x.load_state_dict(state_dict)
        self.attr_feature1_x = nn.Sequential(*list(self.attr_model_x.children())[:5])
        self.attr_feature2_x = list(self.attr_model_x.children())[5]
        
        self.attr_model_s = torchvision.models.resnet50(pretrained=False)
        self.attr_model_s.load_state_dict(state_dict)
        self.attr_feature1_s = nn.Sequential(*list(self.attr_model_s.children())[:5])
        self.attr_feature2_s = list(self.attr_model_s.children())[5]
        self.attr_feature3_s = list(self.attr_model_s.children())[6]
        self.attr_feature4_s = list(self.attr_model_s.children())[7]

        # ============ 原有的融合模块 ============
        self.up1 = up_conv_bn_relu(up_size=64, in_channels=2048, out_channels=256)
        self.CBR1 = conv_bn_relu(512, 56)
        self.CBR2 = conv_bn_relu(512, 56)
        self.CBR3 = conv_bn_relu(1024, 56)
        self.CBR5 = conv_bn_relu(56, 256)
        self.CBR6 = conv_bn_relu(512, 256)
        self.CBR7 = conv_bn_relu(512, 256)

        self.SADEM1 = SADEM(56, 16)
        self.SADEM2 = SADEM(56, 16)
        self.CBAM = CBAMLayer(256)

        # ============ 属性特征融合模块（共享）============
        self.attr_up1 = up_conv_bn_relu(up_size=64, in_channels=2048, out_channels=256)
        self.attr_CBR1 = conv_bn_relu(512, 56)
        self.attr_CBR2 = conv_bn_relu(512, 56)
        self.attr_CBR3 = conv_bn_relu(1024, 56)
        self.attr_CBR5 = conv_bn_relu(56, 256)
        self.attr_CBR6 = conv_bn_relu(512, 256)
        self.attr_SADEM1 = SADEM(56, 16)
        self.attr_SADEM2 = SADEM(56, 16)

        # ============ SAGA: 视觉-文本融合模块 ============
        # 为每个模态创建视觉-文本融合模块
        self.overall_fusion = SimpleVisualTextFusion(visual_dim=256, text_dim=4096)
        self.brightness_fusion = SimpleVisualTextFusion(visual_dim=256, text_dim=4096)
        self.contrast_fusion = SimpleVisualTextFusion(visual_dim=256, text_dim=4096)
        self.saturation_fusion = SimpleVisualTextFusion(visual_dim=256, text_dim=4096)
        self.hue_fusion = SimpleVisualTextFusion(visual_dim=256, text_dim=4096)
        self.blur_fusion = SimpleVisualTextFusion(visual_dim=256, text_dim=4096)
        
        # ============ SAGA: 多模态最终融合模块 ============
        self.multimodal_fusion = get_fusion_module(
            version=saga_version,
            feature_dim=256,
            output_dim=256,
            sensitivity_csv_path=sensitivity_csv_path
        )

        # ============ 预测头 ============
        # AVA数据集使用10维Softmax输出，其他数据集使用1维Sigmoid
        if dataset == 'AVA':
            self.head = nn.Sequential(
                nn.PReLU(),
                nn.Dropout(p=0.75),
                nn.Linear(256, 128),
                nn.PReLU(),
                nn.Dropout(p=0.75),
                nn.Linear(128, 10),  # AVA: 10维概率分布
                nn.Softmax(dim=1)
            )
            print(f"✅ AVA数据集: 使用10维Softmax输出")
        else:
            self.head = nn.Sequential(
                nn.PReLU(),
                nn.Dropout(p=0.75),
                nn.Linear(256, 128),
                nn.PReLU(),
                nn.Dropout(p=0.75),
                nn.Linear(128, 1),  # 回归任务: 1维Sigmoid
                nn.Sigmoid()
            )
            print(f"✅ {dataset}数据集: 使用1维Sigmoid输出")
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        print(f"✅ EAMBNet-AGREE 初始化完成\n")

    def _process_attribute(self, attr_tensor):
        """
        处理单个属性，返回融合后的特征（到情感分支融入前）
        
        Args:
            attr_tensor: [B, 3, 512, 512] 属性表示 
            
        Returns:
            attr_feature: [B, 256, 64, 64] 属性融合特征
        """
        input_size = attr_tensor.size()
        
        # 下采样用于上下文分支
        attr_s = F.interpolate(attr_tensor, size=[(input_size[2] // 2), (input_size[3] // 2)], 
                              mode="bilinear", align_corners=True)
        
        # 细节分支
        attr_x1 = self.attr_feature1_x(attr_tensor)  # [B, 256, 128, 128]
        attr_x2 = self.attr_feature2_x(attr_x1)      # [B, 512, 64, 64]
        
        # 上下文分支
        attr_s1 = self.attr_feature1_s(attr_s)       # [B, 256, 64, 64]
        attr_s2 = self.attr_feature2_s(attr_s1)      # [B, 512, 32, 32]
        attr_s3 = self.attr_feature3_s(attr_s2)      # [B, 1024, 16, 16]
        attr_s4 = self.attr_feature4_s(attr_s3)      # [B, 2048, 8, 8]
        
        # 特征融合
        attr_s2_proc = self.attr_CBR1(attr_s2)       # [B, 512, 32, 32] → [B, 56, 32, 32]
        attr_x2_proc = self.attr_CBR2(attr_x2)       # [B, 512, 64, 64] → [B, 56, 64, 64]
        attr_C = self.attr_SADEM1(attr_x2_proc, attr_s2_proc)  # [B, 56, 64, 64]
        
        attr_s3_proc = self.attr_CBR3(attr_s3)       # [B, 1024, 16, 16] → [B, 56, 16, 16]
        attr_C = self.attr_SADEM2(attr_C, attr_s3_proc)        # [B, 56, 64, 64]
        attr_C = self.attr_CBR5(attr_C)              # [B, 56, 64, 64] → [B, 256, 64, 64]
        
        # 处理最高级上下文特征
        attr_x4_ = self.attr_up1(attr_s4)            # [B, 2048, 8, 8] → [B, 256, 64, 64]
        attr_cat = torch.cat((attr_C, attr_x4_), dim=1)  # [B, 512, 64, 64]
        attr_cat = self.attr_CBR6(attr_cat)          # [B, 512, 64, 64] → [B, 256, 64, 64]
        
        return attr_cat

    def forward(self, x, brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr,
                overall_text, brightness_text, contrast_text, saturation_text, hue_text, blur_text,
                image_ids=None, return_sensitivity=False):
        """
        前向传播（支持SAGA v1/v2/v3）
        
        Args:
            x: [B, 3, 512, 512] 原图像
            brightness_attr: [B, 3, 512, 512] 亮度属性
            contrast_attr: [B, 3, 512, 512] 对比度属性
            saturation_attr: [B, 3, 512, 512] 饱和度属性
            hue_attr: [B, 3, 512, 512] 色相属性
            blur_attr: [B, 3, 512, 512] 模糊属性
            overall_text: [B, 4096] 整体文本特征
            brightness_text: [B, 4096] 亮度文本特征
            contrast_text: [B, 4096] 对比度文本特征
            saturation_text: [B, 4096] 饱和度文本特征
            hue_text: [B, 4096] 色相文本特征
            blur_text: [B, 4096] 模糊文本特征
            image_ids: list of str, 图像文件名（v2/v3使用）
            return_sensitivity: bool, 是否返回预测的敏感度（v3使用）
            
        Returns:
            score: [B, 1] 美学分数
            pred_sensitivity (optional): [B, 5] 预测的敏感度（v3且return_sensitivity=True时）
        """
        input_size = x.size()
        
        # ============ 原图像处理 ============
        s = F.interpolate(x, size=[(input_size[2] // 2), (input_size[3] // 2)], mode="bilinear", align_corners=True)
        logits, cam, EAM, conf = self.emotion_model(x)
        x1 = self.feature1_x(x)  # 256, 128,128
        x2 = self.feature2_x(x1)  # 512,64,64

        s1 = self.feature1_s(s)  # 256, 64,64
        s2 = self.feature2_s(s1)  # 512,32,32
        s3 = self.feature3_s(s2)  # 1024,16,16
        s4 = self.feature4_s(s3)

        s2 = self.CBR1(s2)
        x2 = self.CBR2(x2)
        C = self.SADEM1(x2, s2)
        s3 = self.CBR3(s3)
        C = self.SADEM2(C, s3)
        C = self.CBR5(C)

        x4_ = self.up1(s4)
        cat = torch.cat((C, x4_), dim=1)
        cat = self.CBR6(cat)  # [B, 256, 64, 64] ← 原图像特征

        # ============ 提取所有属性的视觉特征 ============
        brightness_visual = self._process_attribute(brightness_attr)  # [B, 256, 64, 64]
        contrast_visual = self._process_attribute(contrast_attr)      # [B, 256, 64, 64]
        saturation_visual = self._process_attribute(saturation_attr)  # [B, 256, 64, 64]
        hue_visual = self._process_attribute(hue_attr)               # [B, 256, 64, 64]
        blur_visual = self._process_attribute(blur_attr)             # [B, 256, 64, 64]

        # ============ SAGA第一阶段：视觉特征 + 文本特征融合 ============
        overall_fused = self.overall_fusion(cat, overall_text)                    # [B, 256, 64, 64]
        brightness_fused = self.brightness_fusion(brightness_visual, brightness_text)  # [B, 256, 64, 64]
        contrast_fused = self.contrast_fusion(contrast_visual, contrast_text)          # [B, 256, 64, 64]
        saturation_fused = self.saturation_fusion(saturation_visual, saturation_text)  # [B, 256, 64, 64]
        hue_fused = self.hue_fusion(hue_visual, hue_text)                             # [B, 256, 64, 64]
        blur_fused = self.blur_fusion(blur_visual, blur_text)                         # [B, 256, 64, 64]

        # ============ SAGA第二阶段：多模态特征融合（根据版本不同） ============
        attr_features = [brightness_fused, contrast_fused, saturation_fused, hue_fused, blur_fused]
        
        if self.saga_version == 'v3' and return_sensitivity:
            # v3: 返回敏感度预测
            final_multimodal_feature, pred_sensitivity = self.multimodal_fusion(
                overall_fused, attr_features, image_ids, return_sensitivity=True
            )
        else:
            # v1/v2: 只返回融合特征
            final_multimodal_feature = self.multimodal_fusion(
                overall_fused, attr_features, image_ids
            )

        # ============ 情感分支融入 ============
        h_EAM = final_multimodal_feature * EAM  # 情感信息融入多模态特征
        h_EAM = self.CBAM(h_EAM)
        Fusion_F = torch.cat((h_EAM, final_multimodal_feature), dim=1)  # [B, 512, 64, 64]
        Fusion_F = self.CBR7(Fusion_F)  # [B, 256, 64, 64]
        score_feature = self.avgpool(Fusion_F).view(Fusion_F.size(0), -1)
        score = self.head(score_feature)
        
        # ============ 返回结果 ============
        if self.saga_version == 'v3' and return_sensitivity:
            return score, pred_sensitivity
        else:
            return score
    
    def attribute_parameters(self):
        """返回属性相关的参数（用于分层学习率）"""
        params = []
        # 属性backbone参数
        params += list(self.attr_model_x.parameters())
        params += list(self.attr_model_s.parameters())
        # 属性融合参数
        params += list(self.attr_up1.parameters())
        params += list(self.attr_CBR1.parameters())
        params += list(self.attr_CBR2.parameters())
        params += list(self.attr_CBR3.parameters())
        params += list(self.attr_CBR5.parameters())
        params += list(self.attr_CBR6.parameters())
        params += list(self.attr_SADEM1.parameters())
        params += list(self.attr_SADEM2.parameters())
        return params
    
    def fusion_parameters(self):
        """返回融合模块的参数（用于分层学习率）"""
        params = []
        params += list(self.overall_fusion.parameters())
        params += list(self.brightness_fusion.parameters())
        params += list(self.contrast_fusion.parameters())
        params += list(self.saturation_fusion.parameters())
        params += list(self.hue_fusion.parameters())
        params += list(self.blur_fusion.parameters())
        params += list(self.multimodal_fusion.parameters())
        return params
