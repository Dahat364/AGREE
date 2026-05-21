"""
属性表示数据集 - 返回原图像和五个科学属性表示
"""

import os
from torchvision import transforms
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from scientific_attribute_extractors import extract_all_attributes

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)

class PARADataset(Dataset):
    """
    PARA数据集：返回原图像和五个属性表示
    """
    def __init__(self, path_to_csv, images_path, if_train):
        """
        Args:
            path_to_csv: CSV文件路径
            images_path: 图像文件夹路径
            if_train: 是否为训练模式
        """
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        
        if if_train:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                normalize
            ])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        # PARA数据集：aestheticScore_mean范围1-5，归一化到0-1
        y = np.array([row['aestheticScore_mean'] / 5])
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
        
        return x, y.astype('float32'), brightness_attr, contrast_attr, saturation_attr, hue_attr, blur_attr
    
    def _normalize_attribute(self, attribute_tensor):
        """
        对属性表示应用ImageNet归一化
        
        Args:
            attribute_tensor: [3, 512, 512] 属性表示张量 (0-1范围)
            
        Returns:
            normalized_attr: [3, 512, 512] 归一化后的属性表示
        """
        mean = torch.tensor(IMAGE_NET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGE_NET_STD).view(3, 1, 1)
        
        # 应用ImageNet归一化
        normalized_attr = (attribute_tensor - mean) / std
        
        return normalized_attr


# 测试函数
def test_attribute_dataset():
    """测试属性表示数据集 (PARA数据集)"""
    print("测试属性表示数据集 (PARA)...")
    
    # 配置路径 - PARA数据集
    csv_path = "/home/dmc/12TB_ZHZ68JLF/A_PROJECT_IAA/IAADatasets/PARA/annotation/processed/train.csv"
    images_path = "/home/dmc/12TB_ZHZ68JLF/A_PROJECT_IAA/IAADatasets/PARA/imgs/"
    
    try:
        dataset = PARADataset(csv_path, images_path, if_train=False)
        
        print(f"数据集大小: {len(dataset)}")
        
        # 测试数据加载
        sample = dataset[0]
        print(f"\n样本包含 {len(sample)} 个元素:")
        print(f"  原图像: {sample[0].shape}")
        print(f"  标签: {sample[1].shape}")
        print(f"  亮度属性: {sample[2].shape}")
        print(f"  对比度属性: {sample[3].shape}")
        print(f"  饱和度属性: {sample[4].shape}")
        print(f"  色相属性: {sample[5].shape}")
        print(f"  模糊属性: {sample[6].shape}")
        
        print("\n✅ 数据集测试成功!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("请检查路径配置")

if __name__ == "__main__":
    test_attribute_dataset()
