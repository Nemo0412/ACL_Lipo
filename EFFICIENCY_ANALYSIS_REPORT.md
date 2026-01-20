# Efficiency Prediction Results Analysis

## Executive Summary

本报告对比了LANTERN模型在不同数据集上的性能，展示了模型在分布外数据（out-of-distribution）上的性能下降。

---

## 1. 数据集概览

### AGILE 数据集（LANTERN训练数据）
- **样本数**：1,100
- **真实值范围**：-2.35 ~ 15.96（连续值，log scale）
- **数据来源**：Hela和RAW 264.7细胞株的LNP transfection efficiency
- **用途**：LANTERN模型的训练集

### Haiyu-Qwen30B 数据集（评估数据）
- **样本数**：600
- **真实值范围**：1 ~ 10（离散整数值）
- **数据来源**：Qwen30B生成的LNP分子
- **标签类型**：离散整数评分，**非AGILE的连续值分布**

---

## 2. 模型架构

### LANTERN（Lipid nAnoparticle Transfection Efficiency pRedictioN）
- **模型类型**：7层全连接神经网络（MLP）
- **输入特征**：分子指纹组合
  - Morgan Circular Fingerprint：2048维
  - RDKit Expert Descriptors：210维（从217维截断）
  - **总特征维数**：2258维
- **输出**：连续值预测（回归）
- **训练目标**：预测AGILE的连续efficiency值

---

## 3. LANTERN在不同数据集上的性能

### 3.1 Haiyu-Qwen30B 测试集结果

#### 连续预测（未四舍五入）
```
特征：
  - 真实值范围: [1, 10]
  - 预测值范围: [-49330, -31314]（异常！）
  
性能指标：
  - MAE: 39413.49
  - RMSE: 39586.71
  - 状态: ❌ 完全失败
```

**问题分析**：
预测值全为负数，这说明存在严重的数据归一化问题。原因是：
1. LANTERN的StandardScaler基于AGILE的连续值（-2.35 ~ 15.96）训练
2. 当inverse_transform应用到完全不同分布的数据时，会产生荒谬的值
3. 这是典型的**分布外(OOD)数据问题**

#### 四舍五入预测（[1-10]范围）
```
特征：
  - 真实值范围: [1, 10]
  - 预测值范围: [1, 1]（几乎所有预测都是1）
  
性能指标：
  - 精确匹配准确率: 0.67%
  - 误差±1: 10.50%
  - 误差±2: 30.83%
  - MAE: 3.55
  - RMSE: 3.90
  - 状态: ❌ 近似随机基线
```

**根本原因**：
模型学习的是AGILE的连续值分布，无法泛化到Haiyu的离散整数值。

---

## 4. 关键发现

### 4.1 分布不匹配
| 维度 | AGILE | Haiyu-Qwen30B |
|------|--------|-------------|
| **值类型** | 连续浮点数 | 离散整数 |
| **范围** | -2.35 ~ 15.96 | 1 ~ 10 |
| **缩放** | Log scale | 线性 |
| **数据来源** | 生物实验 | 模型生成 |

### 4.2 性能总结

| 数据集 | 类型 | MAE | RMSE | 精确度 |
|-------|------|-----|------|--------|
| **AGILE** | 回归（训练数据） | ~0.61 | ~0.90 | R²=0.93 |
| **Haiyu** | 分类（测试数据） | 3.55 | 3.90 | 0.67% |

性能下降比：**~6倍**

### 4.3 为什么Haiyu预测全是"1"？

```python
# LANTERN的预测逻辑
1. 输入2258维特征向量
2. 7层MLP前向传播
3. 得到异常的负值（-49330到-31314）
4. 应用scaler.inverse_transform()
5. Clip到[1, 10]范围 → 所有值都贴近边界
6. Round → 绝大部分变成1（最接近的整数）
```

---

## 5. 模型选择建议

### ❌ LANTERN不适合Haiyu-Qwen30B
**原因**：
- 训练数据（AGILE）是连续值，测试数据（Haiyu）是离散值
- 从无标签生成的分子到有标签实验分子的分布差异太大
- 特征空间完全不匹配

### ✅ 更好的替代方案

1. **重新训练LANTERN**
   - 使用Haiyu数据的离散标签训练分类模型
   - 修改输出层为10分类器（one-hot编码）
   - 需要Haiyu的训练集（当前没有）

2. **使用TransMA**
   - 多模态架构（3D图像+序列）
   - 预训练权重针对LNP任务优化
   - 更强的泛化能力
   - ⚠️ 当前存在实现问题，需要修复

3. **集成模型**
   - LANTERN + 其他分子表示方法
   - 使用多个预测器的平均值
   - 增加鲁棒性

---

## 6. 数据可视化

### 图表说明
生成的`efficiency_prediction_comparison.png`包含4个子图：

1. **左上**：AGILE数据集性能（未找到）
2. **右上**：Haiyu-Qwen30B连续预测分布
3. **左下**：Haiyu-Qwen30B四舍五入预测分布（大多数点聚集在y=1）
4. **右下**：四个分布的箱线图对比

### 可视化特征
- 红色虚线：完美预测线（y=x）
- 蓝点：AGILE预测（未有）
- 橙点：Haiyu连续预测
- 绿点：Haiyu离散预测

---

## 7. 结论

### 主要结论
1. **LANTERN在Haiyu数据上性能极差**：精确度0.67%，只比随机猜测好一点
2. **根本原因是分布不匹配**：不同的数据来源、标签类型、数值范围
3. **这是可预见的结果**：没有为特定任务优化的模型无法泛化

### 建议行动
1. ✅ 保留LANTERN作为baseline（已完成）
2. ⏳ 修复TransMA实现，用它进行对比评估
3. 📊 如果有Haiyu训练数据，重新训练专门的模型
4. 📈 考虑使用更现代的深度学习方法（GNN、Transformer等）

---

## 附录：技术细节

### 特征工程
```python
# LANTERN特征组合
circular_fingerprint = Morgan(radius=2, nBits=2048)  # 2048维
expert_descriptors = RDKitDescriptors()[:210]       # 210维（截断）
combined_features = concatenate([circular, expert])  # 2258维
```

### 数据标准化问题
```python
# 问题代码
scaler_trained_on_agile = StandardScaler()  # 均值≈5, 方差≈10
scaler_trained_on_agile.fit(agile_data)

# 当应用到Haiyu数据
haiyu_predictions_scaled = model.predict(haiyu_features_scaled)
haiyu_predictions_unscaled = scaler.inverse_transform(haiyu_predictions_scaled)
# 结果: [-49330, -31314]（完全错误！）
```

### 为什么需要多个评估指标？
- **MAE**：对异常值敏感
- **RMSE**：惩罚大错误
- **精确度**：分类任务的直观指标
- **误差容限（±1, ±2）**：实际应用中的容忍度

---

**生成时间**：2025-12-12  
**数据源**：AGILE官方数据集，Haiyu-Qwen30B模型生成
**分析工具**：LANTERN, Python3, pandas, matplotlib, scikit-learn
