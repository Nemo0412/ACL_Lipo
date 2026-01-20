"""
使用 Qwen/Qwen3-32B 模型预测虚拟库中每个药物的效率分数
"""
import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm
import time

# 配置模型
MODEL = "Qwen/Qwen2.5-32B-Instruct"  # 如果使用 Qwen3-32B，请修改为相应的模型名称
API_BASE = "你的API地址"  # 请填入你的API地址
API_KEY = "你的API密钥"  # 请填入你的API密钥

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

def predict_efficiency(smiles: str, drug_info: dict, max_retries: int = 3) -> float:
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
                temperature=0.7,
                max_tokens=50
            )
            
            # 提取回复内容
            content = response.choices[0].message.content.strip()
            
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
            print(f"预测失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # 等待2秒后重试
            else:
                print(f"达到最大重试次数，使用默认分数 5.0")
                return 5.0  # 默认中等分数
    
    return 5.0

def main():
    """主函数：读取数据，预测效率分数，排序并输出JSON"""
    
    # 读取数据
    print("正在读取数据...")
    df = pd.read_excel('../data/virtual library-2500.xlsx')
    print(f"共有 {len(df)} 个药物需要预测")
    
    # 存储结果
    results = []
    
    # 对每个药物进行预测
    print("\n开始预测效率分数...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="预测进度"):
        drug_info = {
            'Number': row['Number'],
            'Amino acid': row['Amino acid'],
            'Protection': row['Protection'],
            'linker': row['linker'],
            'OCOO': row['OCOO'],
            'tail': row['tail']
        }
        
        smiles = row['smiles']
        
        # 预测效率分数
        efficiency_score = predict_efficiency(smiles, drug_info)
        
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
        
        # 每10个药物保存一次中间结果（防止意外中断）
        if (idx + 1) % 10 == 0:
            with open('../data/efficiency_scores_intermediate.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 按效率分数从高到低排序
    print("\n正在排序...")
    results_sorted = sorted(results, key=lambda x: x['efficiency_score'], reverse=True)
    
    # 输出到JSON文件
    output_file = '../data/efficiency_scores_ranked.json'
    print(f"\n正在保存结果到 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_sorted, f, ensure_ascii=False, indent=2)
    
    print(f"\n完成！结果已保存到 {output_file}")
    print(f"\n统计信息：")
    print(f"- 总药物数: {len(results_sorted)}")
    print(f"- 平均效率分数: {sum(r['efficiency_score'] for r in results_sorted) / len(results_sorted):.2f}")
    print(f"- 最高效率分数: {results_sorted[0]['efficiency_score']} (药物编号: {results_sorted[0]['number']})")
    print(f"- 最低效率分数: {results_sorted[-1]['efficiency_score']} (药物编号: {results_sorted[-1]['number']})")
    
    # 输出前10名药物
    print("\n效率分数前10名药物：")
    for i, drug in enumerate(results_sorted[:10], 1):
        print(f"{i}. 药物编号 {drug['number']}: 效率分数 {drug['efficiency_score']}")

if __name__ == "__main__":
    main()
