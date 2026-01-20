"""
使用改进的多维度 Prompt 预测药物效率分数
从多个维度评分，然后综合计算最终分数
"""
import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import time

# 模型配置
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 测试模式
TEST_MODE = True  # 先测试
TEST_SIZE = 20

print("=" * 70)
print("药物效率分数预测系统 - 多维度评分版本")
print("=" * 70)
print(f"模型: {MODEL_NAME}")
print(f"评分维度: 分子量、脂溶性、极性、稳定性、生物利用度")
print(f"测试模式: {'是' if TEST_MODE else '否'} (测试 {TEST_SIZE} 个)" if TEST_MODE else "处理全部")
print("=" * 70)

# 加载模型
print("\n正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)
print("✓ 模型加载成功！")

def create_multidim_prompt(smiles: str, drug_info: dict) -> str:
    """创建多维度评分的提示词"""
    prompt = f"""你是一位专业的药物化学专家。请从多个维度分析以下药物分子，并为每个维度评分。

药物信息：
- 编号: {drug_info.get('Number', 'N/A')}
- 氨基酸: {drug_info.get('Amino acid', 'N/A')}
- 保护基: {drug_info.get('Protection', 'N/A')}
- 连接体长度: {drug_info.get('linker', 'N/A')}
- OCOO: {drug_info.get('OCOO', 'N/A')}
- 尾部长度: {drug_info.get('tail', 'N/A')}
- SMILES: {smiles}

请从以下5个维度评分（每个维度0-2分，整数）：

1. **分子量适中性** (0-2分)
   - 评估分子大小是否适合药物使用
   - 0分：过大或过小
   - 1分：可接受
   - 2分：理想范围

2. **脂溶性平衡** (0-2分)
   - 评估脂肪族链和极性基团的平衡
   - 0分：极端（过亲水或过疏水）
   - 1分：一般平衡
   - 2分：良好平衡

3. **结构稳定性** (0-2分)
   - 评估化学键的稳定性、酯键数量
   - 0分：易降解
   - 1分：中等稳定
   - 2分：稳定

4. **极性适宜度** (0-2分)
   - 评估极性基团（酯基、酰胺等）的分布
   - 0分：极性不佳
   - 1分：极性可接受
   - 2分：极性良好

5. **生物利用度潜力** (0-2分)
   - 综合考虑氨基酸类型、连接体、保护基
   - 0分：生物利用度差
   - 1分：生物利用度一般
   - 2分：生物利用度好

请按以下格式输出（只输出数字，用逗号分隔）：
分子量,脂溶性,稳定性,极性,生物利用度

例如: 2,1,2,1,2"""
    return prompt

def predict_multidim_score(smiles: str, drug_info: dict) -> dict:
    """使用多维度评分预测"""
    prompt = create_multidim_prompt(smiles, drug_info)
    
    messages = [
        {"role": "system", "content": "你是专业的药物化学专家。请严格按照要求输出5个整数分数，用逗号分隔。"},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=50,
            temperature=0.5,
            top_p=0.9,
            do_sample=True
        )
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    print(f"  模型回复: {response}")
    
    # 解析多维度分数
    import re
    try:
        # 提取数字
        numbers = re.findall(r'\d+', response)
        if len(numbers) >= 5:
            scores = [min(int(n), 2) for n in numbers[:5]]  # 每个维度最高2分
            total_score = sum(scores)  # 总分0-10
            
            return {
                'molecular_weight': scores[0],
                'lipophilicity': scores[1],
                'stability': scores[2],
                'polarity': scores[3],
                'bioavailability': scores[4],
                'total_score': total_score
            }
        else:
            print(f"  ⚠️  无法解析足够的分数，使用默认值")
            return {
                'molecular_weight': 1,
                'lipophilicity': 1,
                'stability': 1,
                'polarity': 1,
                'bioavailability': 1,
                'total_score': 5
            }
    except Exception as e:
        print(f"  ⚠️  解析失败: {e}")
        return {
            'molecular_weight': 1,
            'lipophilicity': 1,
            'stability': 1,
            'polarity': 1,
            'bioavailability': 1,
            'total_score': 5
        }

def main():
    # 读取数据
    print("\n读取数据...")
    df = pd.read_excel('../data/virtual library-2500.xlsx')
    
    if TEST_MODE:
        df = df.head(TEST_SIZE)
        print(f"测试模式: 处理前 {TEST_SIZE} 个药物")
    else:
        print(f"处理全部 {len(df)} 个药物")
    
    results = []
    start_time = time.time()
    
    print("\n开始预测...")
    print("-" * 70)
    
    for idx, row in df.iterrows():
        print(f"\n[{idx + 1}/{len(df)}] 药物 {row['Number']}")
        
        drug_info = {
            'Number': row['Number'],
            'Amino acid': row['Amino acid'],
            'Protection': row['Protection'],
            'linker': row['linker'],
            'OCOO': row['OCOO'],
            'tail': row['tail']
        }
        
        scores = predict_multidim_score(row['smiles'], drug_info)
        
        print(f"  ✓ 维度分数: 分子量={scores['molecular_weight']}, 脂溶性={scores['lipophilicity']}, "
              f"稳定性={scores['stability']}, 极性={scores['polarity']}, 生物利用度={scores['bioavailability']}")
        print(f"  ✓ 总分: {scores['total_score']}/10")
        
        results.append({
            'number': int(row['Number']),
            'amino_acid': str(row['Amino acid']),
            'protection': str(row['Protection']),
            'linker': str(row['linker']),
            'OCOO': str(row['OCOO']),
            'tail': str(row['tail']),
            'smiles': row['smiles'],
            'molecular_weight_score': scores['molecular_weight'],
            'lipophilicity_score': scores['lipophilicity'],
            'stability_score': scores['stability'],
            'polarity_score': scores['polarity'],
            'bioavailability_score': scores['bioavailability'],
            'efficiency_score': scores['total_score']
        })
    
    elapsed = time.time() - start_time
    
    # 排序
    print("\n排序...")
    results_sorted = sorted(results, key=lambda x: x['efficiency_score'], reverse=True)
    
    output_file = '../data/efficiency_scores_multidim.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_sorted, f, ensure_ascii=False, indent=2)
    
    # 统计
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)
    print(f"处理数量: {len(results_sorted)}")
    print(f"总耗时: {elapsed:.1f} 秒")
    print(f"平均: {elapsed/len(results_sorted):.2f} 秒/药物")
    print(f"平均分数: {sum(r['efficiency_score'] for r in results_sorted)/len(results_sorted):.2f}")
    print(f"最高分: {results_sorted[0]['efficiency_score']} (药物 {results_sorted[0]['number']})")
    print(f"最低分: {results_sorted[-1]['efficiency_score']} (药物 {results_sorted[-1]['number']})")
    
    # 分数分布
    from collections import Counter
    score_dist = Counter([r['efficiency_score'] for r in results_sorted])
    print(f"\n分数分布:")
    for score in sorted(score_dist.keys(), reverse=True):
        print(f"  {score}分: {score_dist[score]} 个药物")
    
    print(f"\nTop 10:")
    for i, drug in enumerate(results_sorted[:10], 1):
        print(f"  {i}. 药物 {drug['number']}: 总分 {drug['efficiency_score']} "
              f"(分子量{drug['molecular_weight_score']}, 脂溶性{drug['lipophilicity_score']}, "
              f"稳定性{drug['stability_score']}, 极性{drug['polarity_score']}, 生物利用度{drug['bioavailability_score']})")
    
    print(f"\n已保存到: {output_file}")

if __name__ == "__main__":
    main()
