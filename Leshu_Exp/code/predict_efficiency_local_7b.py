"""
使用本地 Qwen2.5-7B 模型预测药物效率分数（内存优化版本）
"""
import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import time

# 模型配置 - 使用 7B 模型，内存需求更小
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 是否使用测试模式
TEST_MODE = True  # 改为 False 处理全部 2588 个药物
TEST_SIZE = 10

print("=" * 70)
print("药物效率分数预测系统 - 本地 Qwen2.5-7B 模型")
print("=" * 70)
print(f"模型: {MODEL_NAME}")
print(f"设备: {DEVICE}")
print(f"测试模式: {'是' if TEST_MODE else '否'} (测试 {TEST_SIZE} 个)" if TEST_MODE else f"将处理全部药物")
print("=" * 70)

# 加载模型
print("\n正在加载模型...")
print("提示：首次运行会自动从 HuggingFace 下载模型（约 14GB）")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

print("✓ 模型加载成功！")

def create_prompt(smiles: str, drug_info: dict) -> str:
    """创建简洁的提示词，强调直接输出整数"""
    return f"""作为药物化学专家，根据SMILES结构预测药物效率分数(0-10整数)。

SMILES: {smiles}
氨基酸: {drug_info.get('Amino acid', 'N/A')}
保护基: {drug_info.get('Protection', 'N/A')}

评分标准:
0-2: 极低效率
3-4: 较低效率  
5-6: 中等效率
7-8: 较高效率
9-10: 极高效率

综合考虑药物相似性、生物利用度、稳定性、靶点结合能力和毒性风险。

直接输出0-10之间的一个整数，例如: 7"""

def predict_score(smiles: str, drug_info: dict) -> float:
    """使用模型预测效率分数"""
    prompt = create_prompt(smiles, drug_info)
    
    messages = [
        {"role": "system", "content": "你是专业的药物化学专家。请直接输出一个0-10之间的整数分数，不要解释。"},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=30,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    # 解析数字并转换为整数
    import re
    try:
        score = int(float(response))
        return float(max(0, min(10, score)))
    except:
        numbers = re.findall(r'\d+', response)
        if numbers:
            score = int(numbers[0])
            return float(max(0, min(10, score)))
        return 5.0  # 默认值

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
        print(f"[{idx + 1}/{len(df)}] 药物 {row['Number']}: ", end="", flush=True)
        
        drug_info = {
            'Number': row['Number'],
            'Amino acid': row['Amino acid'],
            'Protection': row['Protection'],
            'linker': row['linker'],
            'OCOO': row['OCOO'],
            'tail': row['tail']
        }
        
        score = predict_score(row['smiles'], drug_info)
        print(f"分数 {score:.2f}")
        
        results.append({
            'number': int(row['Number']),
            'amino_acid': str(row['Amino acid']),
            'protection': str(row['Protection']),
            'linker': str(row['linker']),
            'OCOO': str(row['OCOO']),
            'tail': str(row['tail']),
            'smiles': row['smiles'],
            'efficiency_score': int(score)
        })
        
        # 每50个保存一次中间结果
        if (idx + 1) % 50 == 0:
            with open('../data/efficiency_scores_intermediate_local.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    
    elapsed = time.time() - start_time
    
    # 排序并保存
    print("\n排序...")
    results_sorted = sorted(results, key=lambda x: x['efficiency_score'], reverse=True)
    
    output_file = '../data/efficiency_scores_ranked_local.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_sorted, f, ensure_ascii=False, indent=2)
    
    # 统计
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)
    print(f"处理数量: {len(results_sorted)}")
    print(f"总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    print(f"平均: {elapsed/len(results_sorted):.2f} 秒/药物")
    print(f"平均分数: {sum(r['efficiency_score'] for r in results_sorted)/len(results_sorted):.2f}")
    print(f"最高分: {results_sorted[0]['efficiency_score']} (药物 {results_sorted[0]['number']})")
    print(f"最低分: {results_sorted[-1]['efficiency_score']} (药物 {results_sorted[-1]['number']})")
    
    print(f"\nTop 10:")
    for i, drug in enumerate(results_sorted[:10], 1):
        print(f"  {i}. 药物 {drug['number']}: {drug['efficiency_score']}")
    
    print(f"\n已保存到: {output_file}")

if __name__ == "__main__":
    main()
