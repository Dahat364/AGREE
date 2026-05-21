import math
import torch
import torch.nn as nn
from einops import rearrange
import timm

from fusion_modules import SimpleVisualTextFusion, get_fusion_module


class SwinFeatureExtractor(nn.Module):
    def __init__(self, pretrained_cfg_path=None, model_name='swinv2_base_window8_256'):
        super().__init__()
        overlay = None
        if pretrained_cfg_path:
            overlay = {'file': pretrained_cfg_path}
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            pretrained_cfg_overlay=overlay
        )

    def forward(self, x):
        features = self.backbone.forward_features(x)
        if isinstance(features, dict):
            features = features.get('last', list(features.values())[-1])
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if features.dim() == 3:
            b, l, c = features.shape
            h = w = int(math.sqrt(l))
            features = features.view(b, h, w, c)
        features = rearrange(features, 'b h w c -> b c h w')
        return features


class FeatureAdapter(nn.Module):
    def __init__(self, in_channels, out_channels=256, target_size=14):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(size=(target_size, target_size), mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.proj(x)
        x = self.upsample(x)
        return x


class HKDFusion(nn.Module):
    def __init__(self, pretrained_cfg_path=None, target_feature_size=14, visual_dim=256,
                 saga_version='v2', sensitivity_csv_path=None):
        super().__init__()
        
        self.saga_version = saga_version
        self.sensitivity_csv_path = sensitivity_csv_path
        
        print(f"\n{'='*80}")
        print(f"🚀 初始化 HKD-IAA-SAGA-{saga_version}")
        print(f"{'='*80}\n")
        self.backbone = SwinFeatureExtractor(pretrained_cfg_path=pretrained_cfg_path)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            dummy_output = self.backbone(dummy)
            self.backbone_dim = dummy_output.shape[1]

        self.target_feature_size = target_feature_size
        self.visual_dim = visual_dim

        self.overall_adapter = FeatureAdapter(self.backbone_dim, visual_dim, target_feature_size)
        self.attr_names = ['brightness', 'contrast', 'saturation', 'hue', 'blur']
        self.attr_adapters = nn.ModuleDict({
            name: FeatureAdapter(self.backbone_dim, visual_dim, target_feature_size)
            for name in self.attr_names
        })

        self.overall_fusion = SimpleVisualTextFusion(visual_dim=visual_dim, text_dim=4096)
        self.attr_fusions = nn.ModuleDict({
            name: SimpleVisualTextFusion(visual_dim=visual_dim, text_dim=4096)
            for name in self.attr_names
        })

        self.multimodal_fusion = get_fusion_module(
            version=saga_version,
            feature_dim=visual_dim,
            output_dim=visual_dim,
            sensitivity_csv_path=sensitivity_csv_path
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.PReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(visual_dim, 128),
            nn.PReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def _extract_visual_features(self, x):
        return self.backbone(x)

    def forward(self, x, brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr,
                overall_text, brightness_text, contrast_text, saturation_text, hue_text, blur_text,
                image_ids=None, return_sensitivity=False):

        overall_feat = self._extract_visual_features(x)
        overall_visual = self.overall_adapter(overall_feat)

        attr_inputs = {
            'brightness': brightness_attr,
            'contrast': contrast_attr,
            'saturation': saturation_attr,
            'hue': hue_attr,
            'blur': blur_attr
        }
        attr_texts = {
            'brightness': brightness_text,
            'contrast': contrast_text,
            'saturation': saturation_text,
            'hue': hue_text,
            'blur': blur_text
        }

        visual_features = {'overall': overall_visual}
        for name, tensor in attr_inputs.items():
            feat = self._extract_visual_features(tensor)
            visual_features[name] = self.attr_adapters[name](feat)

        fused_features = {
            'overall': self.overall_fusion(overall_visual, overall_text)
        }
        for name in self.attr_names:
            fused_features[name] = self.attr_fusions[name](visual_features[name], attr_texts[name])

        overall_fused = fused_features['overall']
        attr_fused = [
            fused_features['brightness'],
            fused_features['contrast'],
            fused_features['saturation'],
            fused_features['hue'],
            fused_features['blur']
        ]
        
        if self.saga_version == 'v3' and return_sensitivity:
            final_feature, pred_sensitivity = self.multimodal_fusion(
                overall_fused, attr_fused, image_ids, return_sensitivity=True
            )
        else:
            final_feature = self.multimodal_fusion(
                overall_fused, attr_fused, image_ids
            )

        pooled = self.avgpool(final_feature).flatten(1)
        score = self.head(pooled)

        if self.saga_version == 'v3' and return_sensitivity:
            return score, pred_sensitivity
        else:
            return score

