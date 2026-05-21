import os
from torchvision import transforms
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from scientific_attribute_extractors import extract_all_attributes
from text_feature_loader import TextFeatureLoader, extract_image_id_from_filename

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)

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
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
        else:
            self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            normalize])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = np.array([row['mean_response_scaled']/10])
        image_id = row['image_filename']
        image_path = os.path.join(self.images_path, image_id)
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
        
        # 加载对应的文本特征
        image_id_clean = extract_image_id_from_filename(image_id)
        text_features = self.text_feature_loader.load_all_text_features(image_id_clean)
        
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
                image_id_clean)  # 添加image_id用于SAGA敏感度查询
    
    def _normalize_attribute(self, attribute_tensor):
        """
        对属性表示应用ImageNet归一化
        """
        mean = torch.tensor(IMAGE_NET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGE_NET_STD).view(3, 1, 1)
        
        # 应用ImageNet归一化
        normalized_attr = (attribute_tensor - mean) / std
        
        return normalized_attr

# 测试函数
def test_multimodal_dataset():
    """测试多模态数据集"""
    # 配置路径
    csv_path = "/home/dmc/12TB_ZHZ68JLF/A_PROJECT_IAA/IAADatasets/LAPIS/annotation/processed/train.csv"  # 请替换为实际路径
    images_path = "/home/dmc/12TB_ZHZ68JLF/A_PROJECT_IAA/IAADatasets/LAPIS/images/"  # 请替换为实际路径
    text_features_root = "/home/dmc/12TB_ZHZ68JLF/A_PROJECT_IAA/IAADatasets/LAPIS/text_features/"
    
    try:
        dataset = MultimodalPARADataset(csv_path, images_path, text_features_root, if_train=False)
        
        print(f"多模态数据集大小: {len(dataset)}")
        
        # 测试数据加载
        sample = dataset[0]
        print(f"样本包含 {len(sample)} 个元素:")
        print(f"  原图像: {sample[0].shape}")
        print(f"  标签: {sample[1].shape}")
        print(f"  亮度属性: {sample[2].shape}")
        print(f"  对比度属性: {sample[3].shape}")
        print(f"  饱和度属性: {sample[4].shape}")
        print(f"  色相属性: {sample[5].shape}")
        print(f"  模糊属性: {sample[6].shape}")
        print(f"  整体文本特征: {sample[7].shape}")
        print(f"  亮度文本特征: {sample[8].shape}")
        print(f"  对比度文本特征: {sample[9].shape}")
        print(f"  饱和度文本特征: {sample[10].shape}")
        print(f"  色相文本特征: {sample[11].shape}")
        print(f"  模糊文本特征: {sample[12].shape}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        print("请检查路径配置")

if __name__ == "__main__":
    test_multimodal_dataset()
