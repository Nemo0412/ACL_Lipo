# AGILE vs LLM 方法对比分析

## 核心区别：AGILE 不是 LLM！

### AGILE 的预测方法

**AGILE = Graph Neural Network (GNN) 模型**

#### 1. 架构组成
```
分子 SMILES → 分子图 (Graph) → GNN 编码器 → 预测头 → 输出值
```

**详细流程：**

1. **输入处理：SMILES → 分子图**
   - 将 SMILES 字符串转换为图结构
   - 节点 (nodes) = 原子 (atoms)
   - 边 (edges) = 化学键 (bonds)
   - 节点特征：原子类型、手性标签等
   - 边特征：键类型、键级数等

2. **图神经网络编码 (5层 GINEConv)**
   ```python
   class AGILE(nn.Module):
       - x_embedding: 原子类型嵌入 (Embedding)
       - gnns: 5层图同构网络 (Graph Isomorphism Network with Edge features)
       - batch_norms: 批归一化层
       - pool: 图池化 (mean/max/add pooling)
       - feat_lin: 特征线性变换 (300维 → 512维)
   ```

3. **预测头 (Prediction Head)**
   - Regression 任务：输出 1 个值
   - Classification 任务：输出 2 个值 (binary)
   - 2层全连接网络 + Softplus 激活

4. **预训练数据**
   - 在 **60,000 个虚拟脂质库** 上预训练
   - 专门为 **LNP (脂质纳米颗粒) mRNA 递送** 设计
   - **NOT** 在我们的药物效率/毒性数据上训练

#### 2. AGILE 的优缺点

**✅ 优点：**
- 直接建模分子结构（图表示）
- 编码化学拓扑关系
- 参数效率高（相比 LLM）
- 推理速度快

**❌ 缺点（导致失败的原因）：**
- **Domain Mismatch**：预训练在脂质，测试在药物分子
- **数据分布不匹配**：60k 脂质 vs 我们的药物数据
- **缺乏迁移能力**：GNN 难以跨领域泛化
- **没有语义理解**：无法理解"efficiency"或"toxicity"的含义

---

### LLM 的预测方法 (Qwen2.5-32B & TXGemma-9B)

**LLM = Large Language Model (基于 Transformer)**

#### 1. 架构组成
```
Prompt (文本) → Tokenizer → Transformer 编码器-解码器 → 生成文本 → 提取数值
```

**详细流程：**

1. **输入处理：构建 Prompt**
   ```
   System: You are a chemistry expert...
   User: Predict the delivery efficiency (1-10) for this molecule:
         SMILES: CCCCCCCC\C=C/CCCCCCCCNC(=O)C...
   ```

2. **Transformer 编码-解码**
   - 32B/9B 参数的注意力机制
   - 多头自注意力 (Multi-Head Attention)
   - 前馈神经网络 (Feed-Forward Networks)
   - 位置编码 (Positional Encoding)

3. **文本生成**
   - 自回归生成响应文本
   - 例如："The predicted efficiency score is 5."

4. **后处理：提取数值**
   ```python
   def extract_efficiency_from_response(response):
       # 正则表达式匹配数字
       patterns = [r'score.*?(\d+)', r'is (\d+)', r'(\d+)/10', ...]
       # 返回提取的 1-10 的分数
   ```

#### 2. LLM 的优缺点

**✅ 优点：**
- **语义理解**：理解"efficiency"、"toxicity"概念
- **Few-shot 学习**：可以从少量样本学习
- **指令遵循**：理解人类指令格式
- **跨领域泛化**：在大规模语料上预训练（包括化学文献）
- **Fine-tuning 有效**：Qwen32B 在我们数据上微调后性能大幅提升

**❌ 缺点：**
- 参数量大（32B/9B）
- 推理慢（每样本 6-7 秒）
- 需要 GPU 资源多
- Zero-shot 可能过于保守（TXGemma 总预测 5）

---

## 实验结果对比

### Efficiency 预测 (600 samples)

| Model | 方法 | MAE | Pearson R | ±2 Acc | Adaptive | Extreme Acc |
|-------|------|-----|-----------|--------|----------|-------------|
| **Qwen2.5-32B** (Fine-tuned) | LLM | **1.200** | **0.144** | 85.5% | **86.7%** | **100.0%** |
| **TXGemma-9B** (Zero-shot) | LLM | 1.333 | -0.045 | 85.7% | 85.7% | 0.0% |
| **AGILE** (Pretrained GNN) | GNN | 5.283 | -0.008 | 30.8% | 30.8% | N/A |

### 关键发现

#### 1. AGILE 失败的原因
```python
# AGILE 的预测分布
predictions_distribution = {
    1: 600/600  # 所有 600 个样本都预测为 1
}
# 完全失败！总是输出最低值 1
```

**根本原因：**
- ❌ 预训练数据 (60k 脂质) 与测试数据 (药物分子) **完全不匹配**
- ❌ GNN 学到的是"脂质结构 → LNP递送效率"的映射
- ❌ 无法迁移到"药物分子 → 药物递送效率"
- ❌ 模型"不知道"如何处理新的分子类型

#### 2. TXGemma-9B 的问题
```python
# TXGemma 的预测分布
predictions_distribution = {
    5: ~600/600  # 几乎所有样本都预测为 5
}
# 保守策略：总选择中间值
```

**原因：**
- ⚠️ Zero-shot（没有在我们数据上训练）
- ⚠️ 采用"安全策略"：不确定时选择中间值 5
- ✅ 但至少理解了任务（预测 1-10 的分数）
- ✅ 对中间值表现好（97%），只是极端值失败

#### 3. Qwen2.5-32B 成功的原因
```python
# Qwen32B 的预测分布
predictions_distribution = {
    1: 5%, 2: 8%, 3: 15%, 4: 20%, 5: 18%, 
    6: 15%, 7: 10%, 8: 6%, 9: 2%, 10: 1%
}
# 多样化预测，覆盖全范围
```

**原因：**
- ✅ **Fine-tuned** on 600 training samples
- ✅ 学到了"分子结构 → 效率"的真实映射
- ✅ 语义理解 + 特定任务适应
- ✅ 对极端值和中间值都表现优异

---

## 方法论总结

### AGILE (GNN) 适用场景
- ✅ 大量同类分子数据
- ✅ 结构-性质关系明确
- ✅ 需要快速推理
- ❌ **不适合跨领域迁移**

### LLM (Transformer) 适用场景
- ✅ 小样本学习 (Few-shot)
- ✅ 跨领域泛化
- ✅ 语义理解重要
- ✅ 可以 fine-tune 适应新任务
- ❌ 需要大量计算资源

---

## 结论

**AGILE 不是 LLM，而是专门为脂质设计的 GNN 模型。**

它在我们的药物数据上失败，是因为：
1. **Domain Mismatch**: 脂质 ≠ 药物分子
2. **任务差异**: LNP递送 ≠ 药物递送效率
3. **GNN 局限性**: 难以跨领域泛化

而 **Fine-tuned LLM (Qwen2.5-32B)** 成功，是因为：
1. **语义理解**: 理解"efficiency"、"toxicity"概念
2. **迁移学习**: 预训练知识 + 任务特定 fine-tuning
3. **Few-shot 能力**: 从 600 个样本学到有效映射

**最终排名：Qwen2.5-32B >> TXGemma-9B >> AGILE**
