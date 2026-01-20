# 精简版 JSON 文件说明

## 📄 文件信息

**文件名**: `efficiency_scores_rdkit_simplified.json`

**包含字段**:
1. `number` - 药物编号
2. `smiles` - SMILES分子结构
3. `efficiency_score` - 效率分数 (0-10)

## 📊 统计信息

- **总药物数**: 2,588
- **平均分数**: 1.29
- **分数范围**: 0-6
- **分数种类**: 7种

## 📈 分数分布

| 分数 | 数量 | 百分比 |
|------|------|--------|
| 6分  | 1    | 0.0%   |
| 5分  | 19   | 0.7%   |
| 4分  | 101  | 3.9%   |
| 3分  | 255  | 9.9%   |
| 2分  | 638  | 24.7%  |
| 1分  | 796  | 30.8%  |
| 0分  | 778  | 30.1%  |

## 🏆 Top 5 药物

1. **药物 2173**: 6分 ⭐ (唯一最高分)
2. **药物 1**: 5分
3. **药物 2**: 5分
4. **药物 14**: 5分
5. **药物 27**: 5分

## 💾 文件大小对比

| 文件类型 | 大小 | 包含字段 |
|---------|------|---------|
| 原始文件 | 1.32 MB | 19个字段（包含详细化学性质） |
| **精简文件** | **0.41 MB** | **3个字段** |
| **减少** | **-68.8%** | - |

## 📝 JSON 格式示例

```json
[
  {
    "number": 2173,
    "smiles": "O=C(OC)C(C(C)O)N(CCC(OCCCCOC(CCCCC)=O)=O)CCC(OCCCCOC(CCCCC)=O)=O",
    "efficiency_score": 6
  },
  {
    "number": 1,
    "smiles": "O=C(OC)CN(CCC(OCCCCOC(CCCCC)=O)=O)CCC(OCCCCOC(CCCCC)=O)=O",
    "efficiency_score": 5
  },
  ...
]
```

## 🔍 使用方法

### Python 读取
```python
import json

with open('efficiency_scores_rdkit_simplified.json', 'r') as f:
    data = json.load(f)

# 获取最高分药物
top_drug = data[0]
print(f"最佳药物: {top_drug['number']}, 分数: {top_drug['efficiency_score']}")

# 筛选特定分数
high_score_drugs = [d for d in data if d['efficiency_score'] >= 5]
print(f"高分药物数量: {len(high_score_drugs)}")
```

### 命令行查看
```bash
# 查看前10个
head -50 efficiency_scores_rdkit_simplified.json

# 统计分数分布
cat efficiency_scores_rdkit_simplified.json | grep "efficiency_score" | sort | uniq -c
```

## 📌 注意事项

1. **已按分数排序**: 文件中的药物已经按 `efficiency_score` 从高到低排序
2. **SMILES 完整性**: 保留了完整的 SMILES 字符串
3. **数据来源**: 基于 RDKit 化学描述符计算（Lipinski's Rule of Five）

## 🎯 关键发现

- **最优候选**: 药物 2173（唯一6分）
- **主要问题**: 大部分药物分子量过大、logP过高
- **合格率**: 只有 4.6% 的药物得分 ≥ 4

---

生成时间: 2025-12-03  
方法: RDKit 化学描述符分析  
处理时间: 2.1 秒 (全部 2588 个药物)
