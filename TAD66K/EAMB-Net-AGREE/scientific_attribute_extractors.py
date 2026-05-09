import torch
import torch.nn.functional as F

def _denorm01(x, normalized, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    """反归一化到[0,1]范围"""
    if not normalized: 
        return x.clamp(0,1)
    device, dtype = x.device, x.dtype
    mean = torch.tensor(mean, device=device, dtype=dtype).view(3,1,1)
    std  = torch.tensor(std,  device=device, dtype=dtype).view(3,1,1)
    return (x*std + mean).clamp(0,1)

def _rgb_to_gray(x):
    """RGB转灰度，使用ITU-R BT.601标准权重"""
    r,g,b = x[0:1], x[1:2], x[2:3]
    return (0.2989*r + 0.5870*g + 0.1140*b)  # [1,H,W]

def _robust01(t):
    """鲁棒归一化：使用1%-99%分位数，避免极值影响"""
    p1, p99 = torch.quantile(t.flatten(), torch.tensor([0.01,0.99], device=t.device))
    t = t.clamp(p1, p99)
    return ((t - p1) / (p99 - p1 + 1e-6)).clamp(0,1)

def _rgb_to_hsv_torch(x):
    """RGB转HSV色彩空间"""
    r, g, b = x[0], x[1], x[2]
    maxc, _ = torch.stack([r,g,b], 0).max(0)
    minc, _ = torch.stack([r,g,b], 0).min(0)
    delta = maxc - minc
    v = maxc
    s = torch.where(maxc > 1e-6, delta / (maxc + 1e-6), torch.zeros_like(maxc))
    
    # 色相计算
    h = torch.zeros_like(maxc)
    mask = delta > 1e-6
    
    # 对不同最大通道分别计算
    r_is_max = (r >= g) & (r >= b) & mask
    g_is_max = (g > r) & (g >= b) & mask
    b_is_max = (b > r) & (b > g) & mask

    h[r_is_max] = ( (g - b)[r_is_max] / (delta[r_is_max] + 1e-6) ) % 6
    h[g_is_max] = ( (b - r)[g_is_max] / (delta[g_is_max] + 1e-6) ) + 2
    h[b_is_max] = ( (r - g)[b_is_max] / (delta[b_is_max] + 1e-6) ) + 4
    h = (h / 6.0) % 1.0  # 归一化到 [0,1)
    return h, s, v

# ============ 五个科学属性提取函数 ============

def extract_brightness_attribute(image_tensor, normalized=False):
    """
    提取亮度属性：使用标准RGB到灰度转换
    
    Args:
        image_tensor: [3,H,W] 输入图像
        normalized: 是否已经ImageNet归一化
        
    Returns:
        brightness_attr: [3,H,W] 亮度属性表示
    """
    x = _denorm01(image_tensor, normalized)          # [3,H,W] in [0,1]
    gray = _rgb_to_gray(x)                           # [1,H,W]
    return gray.repeat(3,1,1)                        # [3,H,W]

def extract_contrast_attribute(image_tensor, normalized=False):
    """
    提取对比度属性：Sobel梯度幅值 + 鲁棒归一化
    
    Args:
        image_tensor: [3,H,W] 输入图像
        normalized: 是否已经ImageNet归一化
        
    Returns:
        contrast_attr: [3,H,W] 对比度属性表示
    """
    x = _denorm01(image_tensor, normalized)
    gray = _rgb_to_gray(x)                           # [1,H,W]
    g = F.pad(gray, (1,1,1,1), mode='reflect')

    device, dtype = x.device, x.dtype
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=device, dtype=dtype).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=device, dtype=dtype).view(1,1,3,3)
    gx = F.conv2d(g, kx)
    gy = F.conv2d(g, ky)
    mag = torch.hypot(gx, gy).squeeze(0).squeeze(0)  # [H,W]
    mag = _robust01(mag)
    return mag.unsqueeze(0).repeat(3,1,1)

def extract_saturation_attribute(image_tensor, normalized=False):
    """
    提取饱和度属性：HSV色彩空间的S通道
    
    Args:
        image_tensor: [3,H,W] 输入图像
        normalized: 是否已经ImageNet归一化
        
    Returns:
        saturation_attr: [3,H,W] 饱和度属性表示
    """
    x = _denorm01(image_tensor, normalized)
    _, s, _ = _rgb_to_hsv_torch(x)                   # [H,W] in [0,1]
    return s.unsqueeze(0).repeat(3,1,1)

def extract_hue_attribute(image_tensor, normalized=False):
    """
    提取色相属性：三相位圆形编码，避免色相环断点问题
    
    Args:
        image_tensor: [3,H,W] 输入图像
        normalized: 是否已经ImageNet归一化
        
    Returns:
        hue_attr: [3,H,W] 色相属性表示
    """
    x = _denorm01(image_tensor, normalized)
    h, s, _ = _rgb_to_hsv_torch(x)                   # [H,W] in [0,1)
    angle = h * 2*torch.pi
    
    # 三相位编码（0°,120°,240°）
    r = (torch.cos(angle) + 1)/2
    g = (torch.cos(angle + 2*torch.pi/3) + 1)/2
    b = (torch.cos(angle + 4*torch.pi/3) + 1)/2
    
    # 用饱和度加权，低饱和度区域色相不重要
    weight = s.unsqueeze(0)
    hue_rgb = torch.stack([r,g,b], dim=0) * weight + 0.5 * (1 - weight)
    
    return hue_rgb.clamp(0,1)

def extract_blur_attribute(image_tensor, normalized=False, return_blur=True):
    """
    提取模糊属性：拉普拉斯响应 + 鲁棒归一化
    
    Args:
        image_tensor: [3,H,W] 输入图像
        normalized: 是否已经ImageNet归一化
        return_blur: True返回模糊度，False返回清晰度
        
    Returns:
        blur_attr: [3,H,W] 模糊属性表示
    """
    x = _denorm01(image_tensor, normalized)
    gray = _rgb_to_gray(x)                           # [1,H,W]
    g = F.pad(gray, (1,1,1,1), mode='reflect')
    device, dtype = x.device, x.dtype
    k = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], device=device, dtype=dtype).view(1,1,3,3)
    resp = torch.abs(F.conv2d(g, k)).squeeze(0).squeeze(0)  # [H,W]
    sharp = _robust01(resp)
    val = 1.0 - sharp if return_blur else sharp
    return val.unsqueeze(0).repeat(3,1,1)

def extract_enhanced_contrast_attribute(image_tensor, normalized=False):
    """
    增强版对比度：结合Sobel梯度和RMS局部对比度
    
    Args:
        image_tensor: [3,H,W] 输入图像
        normalized: 是否已经ImageNet归一化
        
    Returns:
        contrast_attr: [3,H,W] 增强对比度属性表示
    """
    x = _denorm01(image_tensor, normalized)
    gray = _rgb_to_gray(x)
    
    # 1. Sobel梯度对比度
    g = F.pad(gray, (1,1,1,1), mode='reflect')
    device, dtype = x.device, x.dtype
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=device, dtype=dtype).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=device, dtype=dtype).view(1,1,3,3)
    gx, gy = F.conv2d(g, kx), F.conv2d(g, ky)
    grad_contrast = torch.hypot(gx, gy).squeeze(0).squeeze(0)
    
    # 2. RMS局部对比度
    win = 9
    pad = win//2
    g_pad = F.pad(gray, (pad,pad,pad,pad), mode='reflect')
    mu = F.avg_pool2d(g_pad, kernel_size=win, stride=1)
    mu2 = F.avg_pool2d(g_pad*g_pad, kernel_size=win, stride=1)
    rms_contrast = torch.sqrt((mu2 - mu*mu).clamp_min(0)).squeeze(0).squeeze(0)
    
    # 3. 融合两种对比度
    grad_norm = _robust01(grad_contrast)
    rms_norm = _robust01(rms_contrast)
    combined = 0.6 * grad_norm + 0.4 * rms_norm
    
    return combined.unsqueeze(0).repeat(3,1,1)

def extract_all_attributes(image_tensor, normalized=False, enhanced_contrast=False):
    """
    提取所有属性的统一接口
    
    Args:
        image_tensor: [3,H,W] 输入图像
        normalized: 是否已经ImageNet归一化
        enhanced_contrast: 是否使用增强版对比度
        
    Returns:
        attributes: dict 包含所有属性的字典
    """
    contrast_func = extract_enhanced_contrast_attribute if enhanced_contrast else extract_contrast_attribute
    
    return {
        'brightness': extract_brightness_attribute(image_tensor, normalized),
        'contrast':   contrast_func(image_tensor, normalized),
        'saturation': extract_saturation_attribute(image_tensor, normalized),
        'hue':        extract_hue_attribute(image_tensor, normalized),
        'blur':       extract_blur_attribute(image_tensor, normalized, return_blur=True),
        'sharpness':  extract_blur_attribute(image_tensor, normalized, return_blur=False),
    }

# 测试函数
def test_scientific_extractors():
    """测试科学属性提取器"""
    print("测试科学属性提取器...")
    
    # 创建测试图像
    test_image = torch.rand(3, 512, 512)
    
    # 测试所有属性
    attributes = extract_all_attributes(test_image, normalized=False, enhanced_contrast=True)
    
    for attr_name, attr_tensor in attributes.items():
        print(f"{attr_name}: 形状 {attr_tensor.shape}, "
              f"范围 [{attr_tensor.min():.3f}, {attr_tensor.max():.3f}], "
              f"均值 {attr_tensor.mean():.3f}")
    
    print("\n测试归一化输入...")
    # 测试归一化输入
    normalized_image = (test_image - torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)) / torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    attributes_norm = extract_all_attributes(normalized_image, normalized=True)
    
    for attr_name, attr_tensor in attributes_norm.items():
        print(f"{attr_name} (归一化输入): 形状 {attr_tensor.shape}, "
              f"范围 [{attr_tensor.min():.3f}, {attr_tensor.max():.3f}]")

if __name__ == "__main__":
    test_scientific_extractors()