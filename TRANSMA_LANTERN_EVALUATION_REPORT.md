# TransMA + LANTERN 评估总结报告

## 实验目标
使用LANTERN和TransMA两种方法评估Haiyu-Qwen30B中efficiency和toxicity数据的预测性能。

## 1. LANTERN 评估结果 ✅

### 模型信息
- **类型**：7层多层感知机 (MLP) 回归器
- **训练数据**：AGILE LNP efficiency dataset (1100 样本，连续值)
- **特征**：Circular Fingerprint (2048维) + RDKit Expert (210维) = 2258维
- **预训练方式**：在AGILE连续分布上预训练

### Efficiency Task 评估 (600个测试样本)

#### 连续值预测
- **MAE (Mean Absolute Error)**：2.19
- **RMSE (Root Mean Square Error)**：2.93
- **预测范围**：[1.00, 9.98]

#### 整数化预测 (圆整到[1-10])
- **精确准确率 (Exact)**：14.83%
- **误差±1准确率**：40.00%
- **误差±2准确率**：61.33%

#### 按极值/中值分类分析
| 类别 | 样本数 | 占比 | Within±2准确率 |
|------|--------|------|-------------|
| 极端值(1,2,9,10) | 70 | 11.67% | 40.00% |
| 中间值(3-8) | 530 | 88.33% | 64.15% |
| **总体** | **600** | **100%** | **61.33%** |

**关键发现**：
- 中间值预测准确率比极端值高**1.60倍**
- 模型倾向预测中间值，对极端值鲁棒性差

### Toxicity Task 评估 (200个测试样本)

#### 错误做法警告 ⚠️
前期使用了**同一个在efficiency上训练的模型**来预测toxicity，通过简单阈值(0.5)二值化：
- **结果**：56-58% 准确率 ❌
- **问题**：接近随机猜测(56% baseline)，完全无效
- **原因**：
  1. 模型未在toxicity数据上训练
  2. 任务不匹配（回归vs分类）
  3. 特征分布完全不同

**结论**：LANTERN的toxicity结果**不可信**，需要单独训练二分类模型。

---

## 2. TransMA 评估 ❌

### 环境配置
✅ **成功安装**：
- PyTorch 2.2.0 (CPU-only, 因为CUDA不可用)
- Uni-Core (从GitHub源码构建)
- 其他依赖 (rdkit, transformers, einops等)

⚠️ **预训练权重**：
- Uni-Mol权重可用 (mol_pre_no_h_220816.pt, 182MB)
- ChemBERTa模型不可用 (需网络访问HuggingFace)
- 缺少完整的LNP finetuned模型

### 实现挑战

#### 问题1：ChemBERTa Tokenizer不可用
```
NetworkError: Failed to establish a new connection
[Errno 101] Network is unreachable to HuggingFace
```
- 系统无网络访问HuggingFace
- 本地备份的ChemBERTa路径已过期 (/home/wk/...)

#### 问题2：Mamba API设计问题
```python
# 错误：Mamba.forward() got an unexpected keyword argument 'net_input'
TypeError: Mamba.forward() got an unexpected keyword argument 'net_input'
```
- 模型API与trainer.py的调用方式不匹配
- 原始代码有bug，需要大量修改

#### 问题3：数据处理复杂度
- TransMA需要特定的数据格式（batch_collate_fn返回字典）
- Uni-Mol要求3D坐标（34.67%分子无法生成3D构象）
- Mamba需要SMILES tokenization

### 评估结果
**无法完成**：TransMA代码存在多个bug，无法直接使用原始权重进行推理或训练。

---

## 3. 模型对比分析

### 架构对比
| 特性 | LANTERN | TransMA |
|------|---------|---------|
| **模型类型** | MLP (7层) | 多模态深度学习 |
| **3D结构编码** | ✗ (仅fingerprint) | ✓ (Uni-Mol) |
| **序列建模** | ✗ | ✓ (Mamba SSM) |
| **文本编码** | ✗ | ✓ (ChemBERTa) |
| **模型复杂度** | 低 | 高 |
| **计算效率** | 快 | 慢 |
| **可解释性** | 高 | 低 |

### 性能对比
| 指标 | LANTERN | TransMA |
|------|---------|---------|
| **Efficiency±2准确率** | 61.33% | ❌ 未完成 |
| **Toxicity准确率** | 56% (无效) | ❌ 未完成 |
| **实现难度** | 简单 | 困难 |
| **依赖完整性** | ✅ 完整 | ❌ 缺少权重/有bug |

---

## 4. 关键发现

### LANTERN的优势
1. ✅ **完整的预训练模型**：可直接使用推理
2. ✅ **稳定的实现**：代码清晰，bug少
3. ✅ **快速推理**：MLP模型计算快
4. ✅ **可靠的结果**：61.33%的±2准确率在医药化学中可接受

### LANTERN的局限
1. ❌ **仅用fingerprint**：无法捕捉3D空间信息
2. ❌ **分布不匹配**：训练数据(连续)vs测试数据(离散)
3. ❌ **任务单一**：需要为toxicity单独训练模型
4. ❌ **黑盒性**：MLP的特征学习过程不可解释

### TransMA的理想优势
1. ✓ **多模态**：结合3D结构、序列和文本信息
2. ✓ **可扩展**：支持多种分子表示
3. ✓ **最新方法**：使用Mamba等新颖架构

### TransMA的实际困境
1. ❌ **代码不稳定**：多个bug需要修复
2. ❌ **依赖不完整**：缺少关键权重和网络访问
3. ❌ **复杂度高**：难以部署和维护
4. ❌ **文档不足**：API设计不清晰

---

## 5. 建议方案

### 立即可用
**使用LANTERN结果** (61.33% Within±2)
- 对于Efficiency：已有可靠的评估结果
- 对于Toxicity：建议单独训练LANTERN的二分类模型

### 中期改进
1. **获取完整TransMA模型**：
   - 联系原作者获得官方预训练权重
   - 或从GitHub release下载预训练模型

2. **修复TransMA代码**：
   - 修复Mamba API不兼容问题
   - 解决ChemBERTa tokenizer缺失

3. **更好的训练数据**：
   - 用AGILE + Haiyu的混合数据训练
   - 使用curriculum learning适应分布转移

### 长期方案
1. **集成学习**：结合LANTERN + TransMA的预测
2. **数据增强**：用GAN生成更多LNP样本
3. **主动学习**：选择模型不确定的样本进行标注

---

## 6. 最终结论

### 数据质量评估
- **Efficiency数据**：质量良好，离散分布[1-10]，600样本足够
- **Toxicity数据**：质量一般，类别不平衡(44% positive)，200样本偏少

### 模型适用性
| 数据 | 最佳方案 | 准确率 | 建议 |
|------|---------|--------|------|
| **Efficiency** | LANTERN (±2) | 61.33% | ✅ 可用于初步筛选 |
| **Toxicity** | 待训练 | - | ⏳ 需要专门模型 |

### 下一步行动
1. **立即**：报告LANTERN效率评估结果
2. **短期**：为Toxicity训练LANTERN二分类模型
3. **中期**：修复或替换TransMA实现
4. **长期**：积累更多高质量的LNP-毒性关联数据

---

## 7. 附录：技术细节

### LANTERN特征工程
```
输入SMILES → RDKit → Molecular Graph
  ↓
Circular Fingerprint: 2048维 (radius=2, nBits=2048)
RDKit Descriptors: 217维 → 截断为210维
  ↓
合并特征：2258维向量
  ↓
StandardScaler 归一化
  ↓
MLP (2258→200→300→500→500→300→200→1) 回归
  ↓
Inverse Transform → [1, 10] 范围
```

### TransMA架构 (未完成)
```
SMILES → [Uni-Mol 3D编码] → 512维
        ↓
         [Mamba序列编码] → 512维
        ↓
         [ChemBERTa文本编码] → 768维
        ↓
         特征融合 → 最终预测
```

---

**报告生成时间**：2025-12-12
**环境**：Python 3.12, PyTorch 2.2.0, CPU-only
**数据量**：Efficiency (600), Toxicity (200)
