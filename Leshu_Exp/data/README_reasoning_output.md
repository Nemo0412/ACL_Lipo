# 药物效率分数预测结果 - JSON 文件说明

## 📁 输出文件

已生成 **2 个 JSON 文件**，包含前 10 个药物的预测结果：

### 1️⃣ 原始文件（推荐使用）
**文件路径**: `/mnt/3fs/dots-pretrain/leshu/workspace/Leshu_Exp/data/efficiency_scores_rdkit_with_reasoning.json`

**文件大小**: 4.9 KB

**格式**: 数组格式，直接包含所有药物信息

**结构**:
```json
[
  {
    "number": 1,
    "smiles": "O=C(OC)CN(CCC(OCCCCOC(CCCCC)=O)=O)...",
    "efficiency_score": 7.5,
    "molecular_weight": 573.7,
    "logP": 4.1,
    "hbd": 0,
    "hba": 11,
    "tpsa": 134.7,
    "rotatable_bonds": 26,
    "reasoning": "由于这个分子具有较高的分子量(573.7 Da)和较多的可旋转键(26)，所以其细胞膜穿透能力和口服生物利用度受到影响，但其他性质尚可，因此评分为7.5/10。"
  },
  ...
]
```

### 2️⃣ 格式化文件（带摘要）
**文件路径**: `/mnt/3fs/dots-pretrain/leshu/workspace/Leshu_Exp/data/efficiency_scores_rdkit_with_reasoning_formatted.json`

**文件大小**: 4.6 KB

**格式**: 对象格式，包含摘要信息和药物列表

**结构**:
```json
{
  "summary": {
    "total_drugs": 10,
    "processing_method": "RDKit化学描述符 + Qwen2.5-32B LLM生成理由",
    "score_range": "0-10（保留一位小数）",
    "reasoning_format": "由于这个分子具有...，所以...",
    "test_size": 10
  },
  "score_distribution": {
    "7.5": {"count": 1, "percentage": 10.0},
    "7.0": {"count": 1, "percentage": 10.0},
    "6.0": {"count": 1, "percentage": 10.0},
    "5.5": {"count": 7, "percentage": 70.0}
  },
  "drugs": [...]
}
```

## 📊 结果统计

- **测试数量**: 10 个药物
- **处理时间**: 61.81 秒
- **平均速度**: 6.18 秒/药物
- **成功率**: 100%

### 分数分布

| 分数 | 数量 | 百分比 |
|------|------|--------|
| 7.5  | 1    | 10.0%  |
| 7.0  | 1    | 10.0%  |
| 6.0  | 1    | 10.0%  |
| 5.5  | 7    | 70.0%  |

## 📝 字段说明

### 每个药物包含的字段：

- **number**: 药物编号
- **smiles**: 分子的 SMILES 结构式
- **efficiency_score**: 效率分数（0-10，保留一位小数）
- **molecular_weight**: 分子量（Da）
- **logP**: 脂溶性（油水分配系数）
- **hbd**: 氢键供体数
- **hba**: 氢键受体数
- **tpsa**: 拓扑极性表面积（Ų）
- **rotatable_bonds**: 可旋转键数
- **reasoning**: 预测理由（LLM 生成，"由于...所以..."格式）

## ✅ Reasoning 格式验证

所有 10 个药物的 reasoning 均符合要求：
- ✓ 以"由于"开头
- ✓ 包含"所以"连接词
- ✓ 说明具体的化学描述符特征
- ✓ 解释分数的依据

### Reasoning 示例：

1. **药物 1（7.5分）**: 由于这个分子具有较高的分子量(573.7 Da)和较多的可旋转键(26)，所以其细胞膜穿透能力和口服生物利用度受到影响，但其他性质尚可，因此评分为7.5/10。

2. **药物 3（6.0分）**: 由于这个分子具有较高的分子量（629.8 Da）、过高的脂溶性（logP=5.7）以及过多的氢键受体（11个），所以其在口服吸收和细胞渗透方面存在较大问题，给予6.0/10的评分。

3. **药物 10（5.5分）**: 由于这个分子具有过大的分子量（826.2 Da）、极高的脂溶性（logP=11.2）以及过多的可旋转键（44），所以其在口服吸收、细胞渗透及代谢稳定性方面存在显著问题，效率评分为5.5/10。

## 🔬 评分方法

### RDKit 化学描述符计算（客观评分）
基于 Lipinski's Rule of Five（类药五原则）计算分数：
- 分子量: 160-480 Da（理想）
- 脂溶性 (logP): 0-5（理想）
- 氢键供体: ≤5（理想）
- 氢键受体: ≤10（理想）
- 极性表面积: <140 Ų（理想）
- 可旋转键: <10（理想）

### Qwen2.5-32B LLM 生成理由（主观解释）
- 模型: Qwen/Qwen2.5-32B-Instruct
- 离线模式: HF_HUB_OFFLINE=1
- 生成方式: 基于化学描述符，解释分数的科学依据
- 输出格式: "由于...所以..."

## 🚀 使用方法

### Python 读取
```python
import json

# 读取原始文件
with open('efficiency_scores_rdkit_with_reasoning.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 查看第一个药物
drug = data[0]
print(f"药物 {drug['number']}")
print(f"分数: {drug['efficiency_score']}")
print(f"理由: {drug['reasoning']}")
print(f"分子量: {drug['molecular_weight']} Da")
print(f"logP: {drug['logP']}")
```

### 筛选高分药物
```python
high_score_drugs = [d for d in data if d['efficiency_score'] >= 7.0]
print(f"高分药物数量: {len(high_score_drugs)}")
```

### 分析化学性质
```python
# 统计分子量分布
weights = [d['molecular_weight'] for d in data]
print(f"平均分子量: {sum(weights)/len(weights):.1f} Da")
```

## 📌 注意事项

1. **当前仅为测试**: 只处理了前 10 个药物
2. **完整处理**: 如需处理全部 2588 个药物，预计需要约 4.4 小时
3. **格式一致性**: 所有 reasoning 已验证符合"由于...所以..."格式
4. **分数保留**: 效率分数保留一位小数（如 7.5、6.0）

## 📖 相关文件

- 原始数据: `virtual library-2500.xlsx`
- 预测脚本: `../code/predict_efficiency_rdkit_with_reasoning.py`
- 日志文件: `../code/llm_test_10_new.log`

---

生成时间: 2025-12-04  
方法: RDKit 化学描述符 + Qwen2.5-32B LLM  
处理时间: 61.81 秒（10 个药物）
