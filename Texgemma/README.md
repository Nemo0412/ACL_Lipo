# TXGemma-27B Virtual Library Prediction

使用 Google TXGemma-27B-Predict 模型预测 10,000 个虚拟分子的 mRNA 转染效率。

## 📁 项目结构

```
Texgemma/
├── data/
│   └── 10000-virtual library.xlsx    # 输入：虚拟分子库
├── predict_virtual_library.py         # 主预测脚本
├── start_prediction.sh                # 启动脚本
├── monitor_prediction.sh              # 监控脚本
├── prediction_log.txt                 # 运行日志
├── result.json                        # 输出：预测结果（按分数排序，含理由）
├── virtual_library_predictions_with_rationale.json  # 完整结果
└── top100_predictions.json            # Top 100 分子
```

## 🚀 快速开始

### 1. 启动预测

```bash
cd /mnt/3fs/dots-pretrain/leshu/workspace/Texgemma
bash start_prediction.sh
```

### 2. 监控进度

```bash
# 实时查看日志
tail -f prediction_log.txt

# 或使用监控脚本
bash monitor_prediction.sh
```

### 3. 查看结果

```bash
# 查看 Top 10 分子
python3 -c "import json; data=json.load(open('result.json')); [print(f\"#{i+1} Score={m['efficiency_score']}: {m['SMILES'][:60]}...\") for i,m in enumerate(data[:10])]"
```

## 📊 输出格式

### result.json (主要结果文件)

按照 efficiency_score 从大到小排序，包含每个分子的预测分数和详细理由：

```json
[
  {
    "ID": 12345,
    "SMILES": "CCCCCCCC...",
    "efficiency_score": 9,
    "rationale": "This molecule exhibits excellent transfection efficiency due to several favorable structural features:\n\n1. Hydrophobic tail length: The molecule contains optimal C16-C18 alkyl chains that provide balanced hydrophobicity...\n\n2. Ester bonds: The strategically positioned ester bonds facilitate biodegradability...\n\n3. Tertiary amines: The presence of 2 tertiary amines enables effective proton buffering..."
  },
  ...
]
```

### 字段说明

- **ID**: 分子编号
- **SMILES**: 分子的 SMILES 表示
- **efficiency_score**: 预测的 mRNA 转染效率分数 (1-10)
  - 1-3: 低效率
  - 4-6: 中等效率
  - 7-8: 良好效率
  - 9-10: 优秀效率
- **rationale**: 详细的预测理由，解释以下因素的影响：
  - Hydrophobic tail length (疏水尾链长度)
  - Ester bond type and position (酯键类型和位置)
  - Number of lipid tails (脂质尾链数量)
  - Number of tertiary amines (叔胺数量)

## ⚙️ 模型信息

- **模型**: google/txgemma-27b-predict (27B 参数)
- **备选**: google/gemma-2-27b-it (如果 txgemma-27b-predict 不可用)
- **推理设置**:
  - Temperature: 0.7
  - Top-p: 0.9
  - Max new tokens: 512
  - Precision: bfloat16

## 📝 Prompt 设计

系统使用以下 prompt 来确保模型提供有理有据的预测：

```
Please predict the mRNA transfection efficiency for the following lipid molecule 
and provide a detailed rationale.

Please provide:
1. An efficiency score from 1 to 10
2. A detailed explanation addressing:
   - Hydrophobic tail length
   - Ester bond type and position
   - Number of lipid tails
   - Number of tertiary amines
   - Overall molecular structure and balance

Format:
Efficiency Score: [1-10]
Rationale: [Detailed explanation]
```

## ⏱️ 预计运行时间

- **总分子数**: 10,000
- **单分子预测时间**: ~10 秒
- **总预计时间**: ~27-28 小时
- **推荐**: 在后台运行，使用 `nohup` 或 `screen`

## 📈 进度监控

脚本会在处理过程中定期输出：

```
[100/10000] Processed 100 molecules
  Last prediction: ID=12345, Score=7
  Rationale preview: This molecule shows good transfection efficiency...

[200/10000] Processed 200 molecules
  ...
```

## 🔍 结果分析

预测完成后，脚本会自动输出：

1. **分数分布统计**
   ```
   Score 10: 45 molecules (0.5%)
   Score 9: 312 molecules (3.1%)
   Score 8: 890 molecules (8.9%)
   ...
   ```

2. **Top 10 分子详情**
   ```
   #1 - ID: 7234
     Score: 10/10
     SMILES: CCCCCCCCCCCCCCCCNC(=O)...
     Rationale: This molecule demonstrates optimal characteristics...
   ```

3. **三个输出文件**
   - `result.json`: 简化版（分数 + 理由）
   - `virtual_library_predictions_with_rationale.json`: 完整版（包含原始响应）
   - `top100_predictions.json`: Top 100 高效分子

## 🛠️ 故障排查

### 模型加载失败

```bash
# 检查 HuggingFace token
cat ~/.cache/huggingface/token

# 手动下载模型
huggingface-cli download google/txgemma-27b-predict --local-dir /tmp/txgemma-27b
```

### 内存不足

```bash
# 检查 GPU 内存
nvidia-smi

# 如果内存不足，可以修改 predict_virtual_library.py 中的 dtype:
# torch.bfloat16 -> torch.float16 (节省内存)
```

### 预测中断

```bash
# 检查最后处理的分子
tail -100 prediction_log.txt

# 脚本会保存中间结果，可以手动恢复
```

## 📚 参考资料

- [TXGemma Model Card](https://huggingface.co/google/txgemma-27b-predict)
- [LNP 设计原理](https://www.nature.com/articles/s41467-024-50619-z)
- [mRNA 转染效率影响因素](https://www.nature.com/articles/nrd.2017.243)

## 💡 使用建议

1. **批量预测**: 当前脚本一次处理所有 10,000 个分子
2. **结果验证**: 建议对 Top 100 高分分子进行实验验证
3. **分数阈值**: 通常 score ≥ 7 的分子值得进一步研究
4. **理由分析**: 关注模型提到的关键结构特征

## 📧 联系方式

如有问题，请联系项目团队。

---

**最后更新**: December 8, 2025
