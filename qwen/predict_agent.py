"""
使用本地 Qwen 模型预测虚拟库中每个药物的效率分数
无需 API，直接加载开源模型进行推理
"""
import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import time
import os

# 模型配置
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"  # 可以改为 Qwen/Qwen2.5-7B-Instruct 如果内存不足
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 测试模式：只处理前N个药物
TEST_MODE = False  # 处理全部 2588 个药物
TEST_SIZE = 5

# 预测参数
TEMPERATURE = 0.3  # 降低温度以获得更一致的预测
MAX_NEW_TOKENS = 50
TOP_P = 0.9

print("=" * 60)
print("药物效率分数预测系统 - 本地模型版本")
print("=" * 60)
print(f"模型: {MODEL_NAME}")
print(f"设备: {DEVICE}")
print(f"测试模式: {'是' if TEST_MODE else '否'}")
if TEST_MODE:
    print(f"测试数量: 前 {TEST_SIZE} 个药物")
print("=" * 60)

# 加载模型和分词器
print("\n正在加载模型和分词器...")
print("(首次运行会从 HuggingFace 下载模型，可能需要一些时间)")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("✓ 模型加载成功！")
    
except Exception as e:
    print(f"\n❌ 模型加载失败: {e}")
    print("\n建议:")
    print("1. 如果是内存不足，尝试使用更小的模型: Qwen/Qwen2.5-7B-Instruct")
    print("2. 如果是网络问题，设置 HF_ENDPOINT 环境变量")
    print("3. 如果需要登录，运行: huggingface-cli login")
    exit(1)

def create_efficiency_prompt(smiles: str, drug_info: dict) -> str:
    """
    创建用于预测药物效率分数的提示词
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

评分标准（0-10分整数）：
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

请直接输出一个0-10之间的整数作为效率分数。只需要输出一个整数，例如：7"""
    
    return prompt

def predict_efficiency(smiles: str, drug_info: dict) -> float:
    """
    使用本地模型预测单个药物的效率分数
    """
    prompt = create_efficiency_prompt(smiles, drug_info)
    
    try:
        # 构建消息
        messages = [
            {"role": "system", "content": "你是一位专业的药物化学专家，能够准确评估药物分子的效率。请直接输出一个0-10之间的整数分数，不要输出任何解释或思考过程。"},
            {"role": "user", "content": prompt}
        ]
        
        # 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码输入
        model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
        
        # 生成回复
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True
            )
        
        # 解码输出
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"  模型回复: {response.strip()}")
        
        # 解析数字并转换为整数
        import re
        response_clean = response.strip()
        
        # 尝试直接转换
        try:
            score = int(float(response_clean))
            score = max(0, min(10, score))
            return float(score)
        except ValueError:
            # 从文本中提取数字
            numbers = re.findall(r'\d+', response_clean)
            if numbers:
                score = int(numbers[0])
                score = max(0, min(10, score))
                return float(score)
            else:
                print(f"  ⚠️  无法从回复中提取数字，使用默认分数 5")
                return 5.0
    
    except Exception as e:
        print(f"  ⚠️  预测失败: {e}")
        return 5.0

def main():
    """主函数：读取数据，预测效率分数，排序并输出JSON"""
    
    # 读取数据
    print("\n正在读取数据...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_excel(os.path.join(base_dir, 'data', 'virtual_library.xlsx'))
    print(f"数据集总数: {len(df)} 个药物")
    
    if TEST_MODE:
        df = df.head(TEST_SIZE)
        print(f"测试模式: 只处理前 {TEST_SIZE} 个药物")
    
    # 存储结果
    results = []
    
    # 对每个药物进行预测
    print(f"\n开始预测效率分数...")
    print("-" * 60)
    
    start_time = time.time()
    
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
            'efficiency_score': int(efficiency_score)  # 保存为整数
        }
        results.append(result_entry)
    
    elapsed_time = time.time() - start_time
    print("\n" + "-" * 60)
    
    # 按效率分数从高到低排序
    print("\n正在排序...")
    results_sorted = sorted(results, key=lambda x: x['efficiency_score'], reverse=True)
    
    # 输出到JSON文件
    output_file = os.path.join(base_dir, 'data', 'qwen_predict_results.json')
    print(f"正在保存结果到 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_sorted, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("预测完成！")
    print("=" * 60)
    print(f"\n统计信息：")
    print(f"- 处理药物数: {len(results_sorted)}")
    print(f"- 总耗时: {elapsed_time:.2f} 秒")
    print(f"- 平均每个药物: {elapsed_time/len(results_sorted):.2f} 秒")
    print(f"- 平均效率分数: {sum(r['efficiency_score'] for r in results_sorted) / len(results_sorted):.2f}")
    print(f"- 最高效率分数: {results_sorted[0]['efficiency_score']} (药物编号: {results_sorted[0]['number']})")
    print(f"- 最低效率分数: {results_sorted[-1]['efficiency_score']} (药物编号: {results_sorted[-1]['number']})")
    
    # 输出排序结果
    print(f"\n效率分数排名（共 {len(results_sorted)} 个）：")
    for i, drug in enumerate(results_sorted[:min(10, len(results_sorted))], 1):
        print(f"  {i}. 药物编号 {drug['number']}: 效率分数 {drug['efficiency_score']}")
    
    if len(results_sorted) > 10:
        print(f"  ... (显示前10个)")
    
    print(f"\n结果已保存到: {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
