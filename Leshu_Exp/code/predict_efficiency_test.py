"""
使用 Qwen/Qwen3-32B 模型预测虚拟库中每个药物的效率分数（测试版 - 前10个药物）
"""
import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm
import time
import os

# 尝试导入配置文件
try:
    from config import API_BASE, API_KEY, MODEL, TEMPERATURE, MAX_TOKENS, MAX_RETRIES, BATCH_SIZE, DELAY_BETWEEN_REQUESTS
except ImportError:
    print("警告: 未找到 config.py 文件，使用默认配置")
    API_BASE = "https://router.huggingface.co/v1"
    API_KEY = os.environ.get("HF_TOKEN", "your-hf-token-here")
    MODEL = "Qwen/Qwen3-32B:groq"
    TEMPERATURE = 0.7
    MAX_TOKENS = 50
    MAX_RETRIES = 3
    BATCH_SIZE = 10
    DELAY_BETWEEN_REQUESTS = 0.5

# 测试模式：只处理前N个药物
TEST_MODE = True
TEST_SIZE = 10

# 初始化客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE
)

def create_efficiency_prompt(smiles: str, drug_info: dict) -> str:
    """
    创建用于预测药物效率分数的提示词
    
    Args:
        smiles: 药物的SMILES表示
        drug_info: 包含药物其他信息的字典
    
    Returns:
        构建好的提示词
    """
    prompt = f"""你是一位专业的药物化学专家，擅长根据分子结构预测药物的效率。

请根据以下药物的SMILES结构和相关信息，预测该药物的效率分数（Efficiency Score）。

药物信息：
- Number: {drug_info.get('Number', 'N/A')}
- Amino acid: {drug_info.get('Amino acid', 'N/A')}
- Protection: {drug_info.get('Protection', 'N/A')}
- Linker: {drug_info.get('linker', 'N/A')}
- OCOO: {drug_info.get('OCOO', 'N/A')}
- Tail: {drug_info.get('tail', 'N/A')}
- SMILES: {smiles}

评分标准（0-10分）：
- 0-2分：效率极低，几乎无活性
- 3-4分：效率较低，活性较弱
- 5-6分：效率中等，有一定活性
- 7-8分：效率较高，活性良好
- 9-10分：效率极高，活性优异

请考虑以下因素进行评分：
1. 分子的药物相似性（Drug-likeness）
2. 分子的生物利用度
3. 分子的稳定性
4. 分子的靶点结合能力
5. 分子的毒性风险

请直接输出一个0-10之间的数字作为效率分数，不要输出其他内容。只需要输出数字，例如：7.5"""
    
    return prompt

def predict_efficiency(smiles: str, drug_info: dict, max_retries: int = MAX_RETRIES) -> float:
    """
    使用模型预测单个药物的效率分数
    
    Args:
        smiles: 药物的SMILES表示
        drug_info: 药物信息字典
        max_retries: 最大重试次数
    
    Returns:
        预测的效率分数（0-10）
    """
    prompt = create_efficiency_prompt(smiles, drug_info)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "你是一位专业的药物化学专家，能够准确评估药物分子的效率。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            
            # 提取回复内容
            content = response.choices[0].message.content.strip()
            print(f"  模型回复: {content}")
            
            # 尝试解析数字
            try:
                score = float(content)
                # 确保分数在0-10之间
                score = max(0.0, min(10.0, score))
                return score
            except ValueError:
                # 如果无法直接转换，尝试从文本中提取数字
                import re
                numbers = re.findall(r'\d+\.?\d*', content)
                if numbers:
                    score = float(numbers[0])
                    score = max(0.0, min(10.0, score))
                    return score
                else:
                    raise ValueError(f"无法从回复中提取数字: {content}")
        
        except Exception as e:
            print(f"  ⚠️  预测失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # 等待2秒后重试
            else:
                print(f"  达到最大重试次数，使用默认分数 5.0")
                return 5.0  # 默认中等分数
    
    return 5.0

def main():
    """主函数：读取数据，预测效率分数，排序并输出JSON"""
    
    print("=" * 60)
    print("药物效率分数预测系统 - 测试模式")
    print("=" * 60)
    print(f"API地址: {API_BASE}")
    print(f"模型: {MODEL}")
    print(f"测试数量: 前 {TEST_SIZE} 个药物")
    print("=" * 60)
    
    # 读取数据
    print("\n正在读取数据...")
    df = pd.read_excel('../data/virtual library-2500.xlsx')
    print(f"数据集总数: {len(df)} 个药物")
    
    if TEST_MODE:
        df = df.head(TEST_SIZE)
        print(f"测试模式: 只处理前 {TEST_SIZE} 个药物")
    
    # 存储结果
    results = []
    
    # 对每个药物进行预测
    print(f"\n开始预测效率分数...")
    print("-" * 60)
    
    for idx, row in df.iterrows():
        print(f"\n[{idx + 1}/{len(df)}] 药物编号: {row['Number']}")
        
        drug_info = {
            'Number': row['Number'],
            'Amino acid': row['Amino acid'],
            'Protection': row['Protection'],
            'linker': row['linker'],
            'OCOO': row['OCOO'],
            'tail': row['tail']
        }
        
        smiles = row['smiles']
        print(f"  SMILES: {smiles[:60]}..." if len(smiles) > 60 else f"  SMILES: {smiles}")
        
        # 预测效率分数
        efficiency_score = predict_efficiency(smiles, drug_info)
        print(f"  ✓ 效率分数: {efficiency_score}")
        
        # 保存结果
        result_entry = {
            'number': int(row['Number']),
            'amino_acid': str(row['Amino acid']),
            'protection': str(row['Protection']),
            'linker': str(row['linker']),
            'OCOO': str(row['OCOO']),
            'tail': str(row['tail']),
            'smiles': smiles,
            'efficiency_score': round(efficiency_score, 2)
        }
        results.append(result_entry)
        
        # 添加延迟避免API限流
        if DELAY_BETWEEN_REQUESTS > 0 and idx < len(df) - 1:
            time.sleep(DELAY_BETWEEN_REQUESTS)
    
    print("\n" + "-" * 60)
    
    # 按效率分数从高到低排序
    print("\n正在排序...")
    results_sorted = sorted(results, key=lambda x: x['efficiency_score'], reverse=True)
    
    # 输出到JSON文件
    output_file = '../data/efficiency_scores_ranked_test.json'
    print(f"正在保存结果到 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_sorted, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print(f"\n统计信息：")
    print(f"- 测试药物数: {len(results_sorted)}")
    print(f"- 平均效率分数: {sum(r['efficiency_score'] for r in results_sorted) / len(results_sorted):.2f}")
    print(f"- 最高效率分数: {results_sorted[0]['efficiency_score']} (药物编号: {results_sorted[0]['number']})")
    print(f"- 最低效率分数: {results_sorted[-1]['efficiency_score']} (药物编号: {results_sorted[-1]['number']})")
    
    # 输出排序结果
    print(f"\n效率分数排名（共 {len(results_sorted)} 个）：")
    for i, drug in enumerate(results_sorted, 1):
        print(f"  {i}. 药物编号 {drug['number']}: 效率分数 {drug['efficiency_score']}")
    
    print(f"\n结果已保存到: {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
