import os
import torch
import numpy as np

class TextFeatureLoader:
    """
    文本特征加载器
    从预提取的文本特征文件中加载对应的特征
    """
    
    def __init__(self, text_features_root):
        """
        Args:
            text_features_root: 文本特征根目录路径
                例如: /home/dmc/12TB_ZHZ68JLF/A_PROJECT_IAA/IAADatasets/PARA/text_features/
        """
        self.text_features_root = text_features_root
        
        # 属性文件夹映射
        self.attribute_folders = {
            'overall': 'overall_features',
            'brightness': 'bright_features',
            'contrast': 'contrast_features', 
            'saturation': 'saturation_features',
            'hue': 'hue_features',
            'blur': 'blur_features'
        }
        
        # 检查文件夹是否存在
        for attr, folder in self.attribute_folders.items():
            folder_path = os.path.join(text_features_root, folder)
            if not os.path.exists(folder_path):
                print(f"警告: 文本特征文件夹不存在: {folder_path}")
    
    def load_text_feature(self, image_id, attribute_name):
        """
        加载指定图像和属性的文本特征
        
        Args:
            image_id: 图像文件名（不含扩展名）
            attribute_name: 属性名称 ('overall', 'brightness', 'contrast', 'saturation', 'hue', 'blur')
            
        Returns:
            text_feature: [1, 4096] 文本特征张量
        """
        # 获取对应的文件夹
        if attribute_name not in self.attribute_folders:
            print(f"警告: 未知属性名称: {attribute_name}")
            return torch.zeros(1, 4096)
            
        folder_name = self.attribute_folders[attribute_name]
        
        # 构建文件路径
        feature_filename = f"{image_id}.pt"
        feature_path = os.path.join(self.text_features_root, folder_name, feature_filename)
        
        try:
            # 加载特征文件
            feature_data = torch.load(feature_path, map_location='cpu')
            
            # 提取特征张量
            if isinstance(feature_data, dict) and 'feature' in feature_data:
                text_feature = feature_data['feature']  # [1, 4096]
            else:
                # 如果是直接的张量
                text_feature = feature_data
                
            # 确保维度正确
            if text_feature.dim() == 1:
                text_feature = text_feature.unsqueeze(0)  # [4096] → [1, 4096]
            elif text_feature.dim() == 2 and text_feature.size(0) != 1:
                text_feature = text_feature[0:1]  # 取第一行
            
            # 🔧 修复：确保数据类型为Float32
            text_feature = text_feature.float()
                
            return text_feature
            
        except Exception as e:
            print(f"加载文本特征失败 {feature_path}: {e}")
            # 返回零特征作为备选（确保Float32类型）
            return torch.zeros(1, 4096, dtype=torch.float32)
    
    def load_all_text_features(self, image_id):
        """
        加载指定图像的所有属性文本特征
        
        Args:
            image_id: 图像文件名（不含扩展名）
            
        Returns:
            text_features: dict 包含所有属性文本特征的字典
                {
                    'overall': [1, 4096],
                    'brightness': [1, 4096],
                    'contrast': [1, 4096],
                    'saturation': [1, 4096],
                    'hue': [1, 4096],
                    'blur': [1, 4096]
                }
        """
        text_features = {}
        
        for attr_name in self.attribute_folders.keys():
            text_features[attr_name] = self.load_text_feature(image_id, attr_name)
            
        return text_features
    
    def batch_load_text_features(self, image_ids, attribute_name):
        """
        批量加载文本特征
        
        Args:
            image_ids: 图像ID列表
            attribute_name: 属性名称
            
        Returns:
            batch_features: [B, 4096] 批量文本特征
        """
        batch_features = []
        
        for image_id in image_ids:
            feature = self.load_text_feature(image_id, attribute_name)
            batch_features.append(feature)
            
        return torch.cat(batch_features, dim=0)  # [B, 4096]

def extract_image_id_from_filename(filename):
    """
    从图像文件名中提取ID（去除扩展名）
    
    Args:
        filename: 图像文件名，例如 'image.jpg'
        
    Returns:
        image_id: 图像ID，例如 'image'
    """
    return os.path.splitext(filename)[0]

# 测试函数
def test_text_feature_loader():
    """测试文本特征加载器 (PARA数据集)"""
    text_features_root = "/home/dmc/12TB_ZHZ68JLF/A_PROJECT_IAA/IAADatasets/PARA/text_features/"
    
    loader = TextFeatureLoader(text_features_root)
    
    # 测试单个特征加载
    test_image_id = "zinaida-serebriakova_turkey-two-odalisques-1916"
    
    print("测试文本特征加载器...")
    print(f"测试图像ID: {test_image_id}")
    
    # 测试各个属性
    for attr_name in ['overall', 'brightness', 'contrast', 'saturation', 'hue', 'blur']:
        feature = loader.load_text_feature(test_image_id, attr_name)
        print(f"{attr_name}: 形状 {feature.shape}, 范围 [{feature.min():.3f}, {feature.max():.3f}]")
    
    # 测试批量加载
    print("\n测试批量加载...")
    all_features = loader.load_all_text_features(test_image_id)
    for attr_name, feature in all_features.items():
        print(f"{attr_name}: {feature.shape}")

if __name__ == "__main__":
    test_text_feature_loader()