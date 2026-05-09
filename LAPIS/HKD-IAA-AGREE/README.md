# HKD-IAA-AGREE - LAPIS

HKD-IAA AGREE version for LAPIS.

> 说明：训练阶段使用 error-aware weighting（warmup + mean=1 归一化），**不计算/不记录/不保存 HMAI 指标**。

## 🚀 用法

### 训练

在本目录下运行：

```bash
python HKDIAA_LAPIS_AGREE_train.py --gpu_id 0 --batch_size 4
```

### 测试

```bash
python HKDIAA_LAPIS_AGREE_test.py
```

## ⚙️ 配置要点（`config.yml`）

- **`checkpoint_dir`**：训练/测试都会优先使用该目录扫描权重
- **`error_aware`**：开启/关闭误差感知加权与 warmup 等参数

## 📊 输出指标

- `ACC / SRCC / PLCC / MAE / MSE / RMSE`
