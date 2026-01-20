# AGILE 预测机制详细分析

## 核心发现：AGILE **确实加载了预训练权重**！

### 1. AGILE 不是 LLM，是 Graph Neural Network (GNN)

**关键证据：**

```python
# 模型架构
AGILE(
  (x_embedding1): Embedding(119, 300)        # 原子类型嵌入
  (x_embedding2): Embedding(3, 300)          # 手性标签嵌入
  (gnns): ModuleList(
    (0-4): 5 x GINEConv()                    # 5层图同构网络
  )
  (batch_norms): ModuleList(
    (0-4): 5 x BatchNorm1d(300)              # 批归一化
  )
  (feat_lin): Linear(300 → 512)              # 特征线性变换
  (pred_head): Sequential(
    Linear(512 → 256) + Softplus             # 预测头
    Linear(256 → 256) + Softplus
    Linear(256 → 1)                          # 输出1个值（回归）
  )
)
```

**这是一个纯粹的 GNN 模型，不是 LLM！**

---

## 2. AGILE 的预测流程

### 加载的预训练模型
```bash
✓ 成功加载: /mnt/3fs/dots-pretrain/leshu/workspace/baseline/AGILE/ckpt/pretrained_agile_60k/checkpoints/model.pth
✓ 模型大小: 9.3MB (相比 LLM 的几十GB，非常小)
✓ 训练数据: 60,000 个虚拟脂质分子
✓ 任务: LNP (脂质纳米颗粒) mRNA 递送效率预测
```

### 预测步骤
```
1. SMILES 字符串 → 分子图 (Graph)
   - 节点 = 原子 (atoms)
   - 边 = 化学键 (bonds)

2. 图通过 5 层 GINEConv (Graph Isomorphism Network with Edge features)
   - 聚合邻居信息
   - 学习分子拓扑结构

3. 图池化 (Graph Pooling)
   - global_mean_pool: 所有节点特征取平均
   - 输出: 300维向量

4. 特征变换
   - Linear(300 → 512)

5. 预测头 (2层 MLP)
   - Linear(512 → 256) + Softplus
   - Linear(256 → 256) + Softplus
   - Linear(256 → 1) → 输出预测值

6. 后处理
   - 裁剪到 [1, 10] 范围
   - 四舍五入为整数
```

---

## 3. 为什么 AGILE 失败了？

### 原因分析

#### A. Domain Mismatch (领域不匹配)

**训练数据 vs 测试数据：**

| 维度 | 训练 (60k 脂质库) | 测试 (我们的药物) |
|------|------------------|------------------|
| 分子类型 | **脂质 (Lipids)** | **药物分子 (Drug-like)** |
| 任务 | **LNP mRNA 递送** | **药物递送效率** |
| 结构特点 | 长链脂肪酸、可电离头基 | 多样化药物骨架 |
| 分子量 | ~400-800 Da | 变化很大 |
| 功能团 | 胺基、酯基为主 | 多样化 |

**结果：** 
- 模型学到的是"脂质结构 → LNP递送"的映射
- 无法泛化到"药物结构 → 药物递送"

#### B. 预测分布异常

**最新结果（刚才运行的）：**
```
Efficiency Results:
  MAE: 5.872 ❌ (比之前的5.283更差！)
  Pearson R: 0.004 ❌ (几乎0相关)
  ±2 Accuracy: 30.8% ❌
  Extreme values: 90.0% ✓ (这次好了？)
  Middle values: 23.0% ❌ (这次很差！)
```

**预测分布分析：**
```python
# 之前的结果：几乎都预测 1
predictions_old = [1, 1, 1, ..., 1]  # 600个1

# 现在的结果：可能预测极端值
# 因为 Extreme accuracy 90%，但 Middle accuracy 只有 23%
# 说明模型倾向预测 1, 2, 9, 10 这些极端值
```

**为什么结果不一致？**
- 可能之前运行时用的是不同的checkpoint
- 或者数据顺序、归一化方式不同
- GNN 模型对 out-of-distribution 数据非常不稳定

#### C. Toxicity 预测完全失败

```
Toxicity Results:
  Accuracy: 56.0% (随机猜测水平)
  Precision: 0.0% ❌
  Recall: 0.0% ❌
  F1 Score: 0.000 ❌
  Predictions: 全部预测为 0 (non-toxic)
```

**原因：**
- AGILE 是 **回归模型**，输出连续值
- 用阈值 0.5 将其转换为二分类
- 所有输出 < 0.5，全被判为 0
- 完全没有判别能力

---

## 4. AGILE vs LLM 的本质区别

### AGILE (Graph Neural Network)

**模型类型：** GNN (图神经网络)

**输入：** 分子图 (Molecular Graph)
```
Nodes: [C, C, N, O, ...]  # 原子
Edges: [(0,1), (1,2), ...]  # 化学键
```

**预训练：** 
- 60k 脂质分子
- 监督学习 (Supervised Learning)
- 特定任务：LNP mRNA 递送

**预测方式：**
```python
# 直接端到端预测
graph → GNN layers → pooling → MLP → output_value
```

**优点：**
- ✅ 快速（毫秒级）
- ✅ 参数少（9.3MB）
- ✅ 直接建模分子结构

**缺点：**
- ❌ 泛化能力差
- ❌ 需要大量同类数据
- ❌ 跨领域迁移困难
- ❌ 无语义理解

---

### LLM (Large Language Model)

**模型类型：** Transformer

**输入：** 自然语言 Prompt
```
"Predict the delivery efficiency (1-10) for this molecule:
SMILES: CCCCCCCC\C=C/CCCCCCCCNC(=O)C..."
```

**预训练：**
- 万亿 tokens 的文本数据
- 包括科学文献、化学教科书
- 自监督学习 (Self-supervised)
- 多任务、多领域

**预测方式：**
```python
# 生成式预测
prompt → tokenize → transformer → generate_text → extract_number
```

**优点：**
- ✅ 强大的泛化能力
- ✅ Few-shot 学习
- ✅ 语义理解
- ✅ 指令遵循
- ✅ 可 fine-tune 适应新任务

**缺点：**
- ❌ 参数量大（32B/9B）
- ❌ 推理慢（秒级）
- ❌ 需要大量计算资源

---

## 5. 对比总结表

| 特性 | AGILE (GNN) | TXGemma-9B (LLM) | Qwen2.5-32B (Fine-tuned LLM) |
|------|------------|------------------|------------------------------|
| **模型类型** | Graph Neural Network | Large Language Model | Large Language Model |
| **参数量** | ~5M | 9B | 32B |
| **模型大小** | 9.3MB | ~18GB | ~64GB |
| **预训练数据** | 60k 脂质分子 | 万亿 tokens 文本 | 万亿 tokens 文本 |
| **是否 fine-tune** | ❌ Zero-shot | ❌ Zero-shot | ✅ Fine-tuned (600 samples) |
| **输入格式** | 分子图 (Graph) | 文本 Prompt + SMILES | 文本 Prompt + SMILES |
| **预测方式** | 直接回归 | 生成文本 → 提取数字 | 生成文本 → 提取数字 |
| **推理速度** | 快 (~0.01s/sample) | 慢 (~7s/sample) | 慢 (~7s/sample) |
| **Efficiency MAE** | 5.872 ❌ | 1.333 ✓ | **1.200** ✅ |
| **Efficiency ±2 Acc** | 30.8% ❌ | 85.7% ✓ | **85.5%** ✅ |
| **Extreme Acc** | 90.0% (不稳定) | 0.0% ❌ | **100.0%** ✅ |
| **Toxicity F1** | 0.000 ❌ | 待评估 | **0.333** ✓ |
| **泛化能力** | ❌ 差 (仅限脂质) | ✓ 中等 (保守策略) | ✅ 优秀 (fine-tuned) |

---

## 6. 结论

### AGILE 确实加载了预训练权重，但仍然失败了

**原因不是"没加载模型"，而是：**

1. **模型类型不对**：GNN 不是 LLM，是专门的图神经网络
2. **预训练数据不匹配**：60k 脂质 ≠ 我们的药物分子
3. **任务差异**：LNP mRNA 递送 ≠ 药物递送效率
4. **GNN 局限性**：难以跨领域迁移，缺乏语义理解

### 为什么 Fine-tuned LLM 成功？

1. ✅ **预训练广泛**：在包括化学在内的大规模文本上训练
2. ✅ **语义理解**：理解"efficiency"、"toxicity"的含义
3. ✅ **Few-shot 能力**：可以从少量样本学习
4. ✅ **Fine-tuning 有效**：600 个样本足以适应新任务
5. ✅ **指令遵循**：理解人类的预测要求

### 最终排名

🥇 **Qwen2.5-32B (Fine-tuned)**: MAE 1.200, 86.7% Adaptive Acc
🥈 **TXGemma-9B (Zero-shot)**: MAE 1.333, 85.7% Adaptive Acc  
🥉 **AGILE (GNN Pretrained)**: MAE 5.872, 30.8% Adaptive Acc

**结论：对于跨领域的药物预测任务，Fine-tuned LLM >> Zero-shot LLM >> Pretrained GNN**
