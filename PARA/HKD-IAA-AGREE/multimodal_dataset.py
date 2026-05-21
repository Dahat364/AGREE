import os
from torchvision import transforms
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from scientific_attribute_extractors import extract_all_attributes
from text_feature_loader import TextFeatureLoader, extract_image_id_from_filename

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
normalize = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)

class MultimodalPARADataset(Dataset):
    """
    多模态PARA数据集：同时返回视觉特征和文本特征
    """
    def __init__(self, path_to_csv, images_path, text_features_root, if_train):
        """
        Args:
            path_to_csv: CSV文件路径
            images_path: 图像文件夹路径
            text_features_root: 文本特征根目录路径
            if_train: 是否为训练模式
        """
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.text_feature_loader = TextFeatureLoader(text_features_root)
        
        if if_train:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                normalize])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        # PARA数据集：aestheticScore_mean范围1-5，归一化到0-1
        y = np.array([row['aestheticScore_mean'] / 5])
        # PARA数据格式：使用imageName和sessionId
        image_id = row['imageName']
        session_id = row['sessionId']
        # PARA使用两层路径结构：sessionId/imageName
        image_path = os.path.join(self.images_path, session_id, image_id)
        image = default_loader(image_path)
        x = self.transform(image)
        
        # 从真实图像中提取科学属性表示
        try:
            attributes = extract_all_attributes(x, normalized=True, enhanced_contrast=True)
            
            brightness_attr = self._normalize_attribute(attributes['brightness'])
            contrast_attr = self._normalize_attribute(attributes['contrast'])
            saturation_attr = self._normalize_attribute(attributes['saturation'])
            hue_attr = self._normalize_attribute(attributes['hue'])
            blur_attr = self._normalize_attribute(attributes['blur'])
            
        except Exception as e:
            # 如果属性提取失败，使用原图像作为备选
            print(f"属性提取失败 {image_id}: {e}")
            brightness_attr = x.clone()
            contrast_attr = x.clone()
            saturation_attr = x.clone()
            hue_attr = x.clone()
            blur_attr = x.clone()
        
        # PARA数据集：文本特征文件名格式为 sessionId__imageName（双下划线）
        image_id_clean = extract_image_id_from_filename(image_id)
        text_feature_id = f"{session_id}__{image_id_clean}"
        text_features = self.text_feature_loader.load_all_text_features(text_feature_id)
        
        # 提取各个属性的文本特征并确保维度正确
        overall_text = text_features['overall'].squeeze(0)        # [1, 4096] → [4096]
        brightness_text = text_features['brightness'].squeeze(0)  # [1, 4096] → [4096]
        contrast_text = text_features['contrast'].squeeze(0)      # [1, 4096] → [4096]
        saturation_text = text_features['saturation'].squeeze(0)  # [1, 4096] → [4096]
        hue_text = text_features['hue'].squeeze(0)               # [1, 4096] → [4096]
        blur_text = text_features['blur'].squeeze(0)             # [1, 4096] → [4096]
        
        return (x, y.astype('float32'), 
                brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr,
                overall_text, brightness_text, contrast_text, saturation_text, hue_text, blur_text,
                image_id)  # 返回imageName用于SAGA敏感度查询
    
    def _normalize_attribute(self, attribute_tensor):
        """
        对属性表示应用ImageNet归一化
        """
        mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
        std = torch.tensor(CLIP_STD).view(3, 1, 1)
        
        # 应用ImageNet归一化
        normalized_attr = (attribute_tensor - mean) / std
        
        return normalized_attr

