# 药物效率分数预测系统

使用 Qwen/Qwen3-32B 大语言模型预测虚拟库中每个药物（SMILES）的效率分数（0-10分），并按分数从高到低排序输出。

## 功能特点

- 🤖 使用 Qwen 大模型进行智能预测
- 📊 支持批量处理 2500+ 个药物分子
- 💾 自动保存中间结果，支持断点续传
- 📈 生成排序后的 JSON 结果文件
- 📉 提供详细的统计信息和分数分布

## 文件说明

### 主要脚本

1. **predict_efficiency.py** - 基础版本
   - 简单直接的实现
   - API配置直接写在代码中
   
2. **predict_efficiency_with_config.py** - 配置文件版本（推荐）
   - 使用外部配置文件
   - 支持断点续传
   - 更详细的统计信息
   - 分数分布分析

3. **config.example.py** - 配置文件模板
   - 复制为 `config.py` 并填入实际配置

## 使用方法

### 1. 安装依赖

```bash
pip install pandas openpyxl openai tqdm
```

### 2. 配置API

复制配置文件模板并填入你的API信息：

```bash
cd code
cp config.example.py config.py
```

编辑 `config.py`，填入以下信息：

```python
API_BASE = "https://your-api-endpoint.com/v1"  # 你的API地址
API_KEY = "your-api-key-here"  # 你的API密钥
MODEL = "Qwen/Qwen2.5-32B-Instruct"  # 模型名称
```

### 3. 运行预测

```bash
cd code
python predict_efficiency_with_config.py
```

## Prompt 设计说明

### 核心 Prompt 结构

本系统使用的 prompt 包含以下几个关键部分：

#### 1. 角色定位
```
你是一位专业的药物化学专家，擅长根据分子结构预测药物的效率。
```

#### 2. 任务描述
明确告诉模型需要根据 SMILES 和药物信息预测效率分数。

#### 3. 输入信息
提供完整的药物信息：
- Number（编号）
- Amino acid（氨基酸）
- Protection（保护基）
- Linker（连接体）
- OCOO（羧基）
- Tail（尾部）
- SMILES（分子结构）

#### 4. 评分标准
明确的 0-10 分评分标准：
- 0-2分：效率极低，几乎无活性
- 3-4分：效率较低，活性较弱
- 5-6分：效率中等，有一定活性
- 7-8分：效率较高，活性良好
- 9-10分：效率极高，活性优异

#### 5. 评估因素
引导模型考虑的关键因素：
1. 分子的药物相似性（Drug-likeness）
2. 分子的生物利用度
3. 分子的稳定性
4. 分子的靶点结合能力
5. 分子的毒性风险

#### 6. 输出格式
明确要求只输出数字，例如：`7.5`

### Prompt 优化建议

如果需要调整 prompt，可以考虑：

1. **增加领域知识**：添加更多药物化学的专业知识
2. **调整评分维度**：根据实际需求调整评估因素
3. **提供示例**：可以添加 few-shot 示例提高准确性
4. **调整温度参数**：在 config.py 中调整 TEMPERATURE（0-1）

## 输出结果

### JSON 文件结构

```json
[
  {
    "number": 1234,
    "amino_acid": "Gly",
    "protection": "Boc",
    "linker": "PEG4",
    "OCOO": "methyl",
    "tail": "C6",
    "smiles": "O=C(OC)CN(CCC(OCCCCOC(CCCCC)=O)=O)...",
    "efficiency_score": 8.5
  },
  ...
]
```

### 输出文件

- **efficiency_scores_ranked.json** - 按效率分数排序的最终结果
- **efficiency_scores_intermediate.json** - 中间结果（处理完成后自动删除）

### 统计信息

运行完成后会显示：
- 总药物数
- 平均效率分数
- 最高/最低效率分数及对应的药物编号
- 效率分数前10名药物
- 分数分布统计

## 功能特性

### 1. 断点续传
如果程序中断，再次运行时会询问是否从上次的进度继续。

### 2. 中间保存
每处理 BATCH_SIZE（默认10个）药物就保存一次中间结果。

### 3. 自动重试
API 调用失败时会自动重试 MAX_RETRIES（默认3次）。

### 4. 速率限制
请求之间添加延迟（DELAY_BETWEEN_REQUESTS），避免触发 API 限流。

### 5. 智能解析
能够从模型输出中智能提取数字分数，即使格式不完全标准。

## 参数配置

在 `config.py` 中可以调整以下参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| TEMPERATURE | 生成温度，控制随机性 | 0.7 |
| MAX_TOKENS | 最大生成token数 | 50 |
| MAX_RETRIES | 失败后最大重试次数 | 3 |
| BATCH_SIZE | 中间保存的批次大小 | 10 |
| DELAY_BETWEEN_REQUESTS | 请求间隔（秒） | 0.5 |

## 注意事项

1. **API 费用**：预测 2588 个药物需要调用 API 2588 次，请注意成本
2. **运行时间**：完整运行可能需要较长时间（取决于 API 响应速度）
3. **网络稳定**：建议在网络稳定的环境下运行
4. **模型选择**：确保使用的模型支持中文和化学领域知识

## 故障排除

### 问题1：API 连接失败
- 检查 API_BASE 和 API_KEY 是否正确
- 确认网络连接正常
- 检查 API 服务是否可用

### 问题2：模型输出格式错误
- 调整 TEMPERATURE 参数（建议 0.5-0.8）
- 检查模型是否支持该 prompt 格式

### 问题3：处理速度慢
- 增加 DELAY_BETWEEN_REQUESTS（但可能触发限流）
- 考虑使用更快的 API 服务
- 使用并行处理（需要修改代码）

## 扩展开发

### 添加并行处理

可以使用 `concurrent.futures` 进行并行处理：

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_batch(drugs):
    results = []
    for drug in drugs:
        score = predict_efficiency(drug['smiles'], drug)
        results.append({**drug, 'efficiency_score': score})
    return results

# 在 main() 中使用
with ThreadPoolExecutor(max_workers=5) as executor:
    # 提交任务并收集结果
    ...
```

### 添加更多评估指标

可以修改 prompt 让模型返回更多信息：

```python
# 除了效率分数，还可以预测：
- 药物相似性分数
- 毒性风险等级
- 建议的改进方向
```

## 许可证

MIT License

## 联系方式

如有问题或建议，请联系开发团队。
