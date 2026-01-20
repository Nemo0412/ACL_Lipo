"""
使用 RDKit 计算分子描述符（保留一位小数）+ Qwen 模型生成预测理由
"""
import pandas as pd
import json
import time
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("=" * 70)
print("药物效率分数预测系统 - RDKit + Qwen 理由生成版本")
print("=" * 70)

# 检查 RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, Crippen
    print("✓ RDKit 已安装")
except ImportError:
    print("❌ 需要安装 RDKit")
    exit(1)

# 配置
TEST_MODE = False  # 处理全部药物
TEST_SIZE = 10
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"

print(f"测试模式: {'是' if TEST_MODE else '否'}")
if TEST_MODE:
    print(f"测试数量: {TEST_SIZE}")
print("=" * 70)

# 加载 Qwen 模型
print("\n正在加载 Qwen 模型...")
import os
os.environ['HF_HUB_OFFLINE'] = '1'  # 强制使用离线模式
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("✓ Qwen 模型加载完成")

def calculate_molecular_descriptors(smiles: str) -> dict:
    """计算分子描述符"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        descriptors = {
            'molecular_weight': round(Descriptors.MolWt(mol), 1),
            'logP': round(Crippen.MolLogP(mol), 1),
            'hbd': Lipinski.NumHDonors(mol),
            'hba': Lipinski.NumHAcceptors(mol),
            'tpsa': round(Descriptors.TPSA(mol), 1),
            'rotatable_bonds': Lipinski.NumRotatableBonds(mol),
        }
        return descriptors
    except Exception as e:
        print(f"错误: {e}")
        return None

def score_molecular_weight(mw: float) -> float:
    """分子量评分 (理想范围: 160-480)"""
    if 160 <= mw <= 480:
        return 2.0
    elif 480 < mw <= 500:
        return 1.5
    elif 140 <= mw < 160 or 500 < mw <= 520:
        return 1.0
    elif 520 < mw <= 600:
        return 0.5
    else:
        return 0.0

def score_lipophilicity(logP: float) -> float:
    """脂溶性评分 (理想范围: 0-5)"""
    if 0 <= logP <= 3:
        return 2.0
    elif 3 < logP <= 5:
        return 1.5
    elif -0.4 <= logP < 0:
        return 1.0
    elif 5 < logP <= 6:
        return 0.5
    else:
        return 0.0

def score_hbond(hbd: int, hba: int) -> float:
    """氢键评分"""
    score = 0.0
    if hbd <= 5:
        score += 1.0
    elif hbd <= 7:
        score += 0.5
    
    if hba <= 10:
        score += 1.0
    elif hba <= 12:
        score += 0.5
    
    return score

def score_tpsa(tpsa: float) -> float:
    """极性表面积评分 (理想: <140)"""
    if tpsa <= 140:
        return 2.0
    elif 140 < tpsa <= 160:
        return 1.0
    elif 160 < tpsa <= 180:
        return 0.5
    else:
        return 0.0

def score_flexibility(rotatable: int) -> float:
    """灵活性评分 (理想: <10)"""
    if rotatable <= 10:
        return 2.0
    elif 10 < rotatable <= 15:
        return 1.0
    elif 15 < rotatable <= 20:
        return 0.5
    else:
        return 0.0

def calculate_efficiency_score(descriptors: dict) -> float:
    """基于 Lipinski's Rule of Five 计算效率分数 (0-10，保留一位小数)"""
    if descriptors is None:
        return 0.0
    
    score = 0.0
    score += score_molecular_weight(descriptors['molecular_weight'])
    score += score_lipophilicity(descriptors['logP'])
    score += score_hbond(descriptors['hbd'], descriptors['hba'])
    score += score_tpsa(descriptors['tpsa'])
    score += score_flexibility(descriptors['rotatable_bonds'])
    
    # 加2后保留一位小数
    final_score = round(score + 2, 1)
    return min(final_score, 10.0)

def generate_reasoning(smiles: str, descriptors: dict, score: float) -> str:
    """使用 Qwen 生成预测理由 - 英文版本"""
    
    # 构建详细的 prompt - 英文
    prompt = f"""You are a medicinal chemistry expert. Explain the efficiency score of this molecule in English.

Molecule SMILES: {smiles}

Chemical Descriptors:
- Molecular Weight: {descriptors['molecular_weight']} Da (ideal: 160-480)
- Lipophilicity (logP): {descriptors['logP']} (ideal: 0-5)
- H-bond Donors: {descriptors['hbd']} (ideal: ≤5)
- H-bond Acceptors: {descriptors['hba']} (ideal: ≤10)
- Topological Polar Surface Area (TPSA): {descriptors['tpsa']} Ų (ideal: <140)
- Rotatable Bonds: {descriptors['rotatable_bonds']} (ideal: <10)

Efficiency Score: {score}/10

Analysis Requirements:
1. Analyze overall chemical descriptors (molecular weight, lipophilicity, etc.)
2. **Focus on head group structure** (e.g., ester, amide functional groups) and its impact on drug efficiency
3. **Focus on tail structure** (e.g., alkyl chains, fatty chains) and its impact on drug efficiency
4. Explain how head and tail structures affect pharmacokinetic properties

Example Format:
"Due to the ester and amide groups at the head and moderate-length fatty chains (C6) at the tail, with molecular weight of 550 Da, the molecule shows good membrane permeability but moderate metabolic stability, resulting in a score of 6.5/10."

Please respond in the format "Due to... (including head and tail structural features), the molecule... (pharmacokinetic effects), resulting in a score of X/10." (80-120 words). Must include specific structural features and numerical values."""

    messages = [
        {"role": "system", "content": "You are a professional medicinal chemistry expert specializing in molecular structure and drug property analysis."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=300,  # 增加长度以容纳头部和尾部结构分析
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

# 主程序
print("\n加载数据...")
df = pd.read_excel('../data/virtual library-2500.xlsx')
print(f"总药物数: {len(df)}")

if TEST_MODE:
    df = df.head(TEST_SIZE)
    print(f"测试模式: 只处理前 {TEST_SIZE} 个药物")

results = []
start_time = time.time()

print("\n开始处理...")
for idx, row in df.iterrows():
    number = row['Number']
    smiles = row['smiles']
    
    # 计算化学描述符
    descriptors = calculate_molecular_descriptors(smiles)
    
    if descriptors is None:
        print(f"跳过药物 {number} (无效 SMILES)")
        continue
    
    # 计算效率分数（保留一位小数）
    score = calculate_efficiency_score(descriptors)
    
    # 生成预测理由
    print(f"处理药物 {number} (分数: {score}) - 生成理由中...")
    reasoning = generate_reasoning(smiles, descriptors, score)
    
    result = {
        'number': int(number),
        'smiles': smiles,
        'efficiency_score': score,
        'molecular_weight': descriptors['molecular_weight'],
        'logP': descriptors['logP'],
        'hbd': descriptors['hbd'],
        'hba': descriptors['hba'],
        'tpsa': descriptors['tpsa'],
        'rotatable_bonds': descriptors['rotatable_bonds'],
        'reasoning': reasoning
    }
    results.append(result)

# 按分数排序
results.sort(key=lambda x: x['efficiency_score'], reverse=True)

# 保存结果到 result.json
output_file = '../data/result.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

elapsed = time.time() - start_time

# 统计信息
print("\n" + "=" * 70)
print("处理完成！")
print("=" * 70)
print(f"总处理时间: {elapsed:.2f} 秒")
print(f"平均每个药物: {elapsed/len(results):.2f} 秒")
print(f"成功处理: {len(results)} 个药物")
print(f"结果已保存到: {output_file}")

# 分数分布
scores = [r['efficiency_score'] for r in results]
score_counts = Counter(scores)
print(f"\n分数分布:")
for score in sorted(score_counts.keys(), reverse=True):
    count = score_counts[score]
    percentage = count / len(results) * 100
    print(f"  {score:.1f}分: {count} 个 ({percentage:.1f}%)")

# 显示前3个结果
print("\n" + "=" * 70)
print("Top 3 药物及预测理由:")
print("=" * 70)
for i, drug in enumerate(results[:3], 1):
    print(f"\n{i}. 药物 {drug['number']} - 分数: {drug['efficiency_score']}/10")
    print(f"   SMILES: {drug['smiles'][:60]}...")
    print(f"   MW: {drug['molecular_weight']}, logP: {drug['logP']}, TPSA: {drug['tpsa']}")
    print(f"   理由: {drug['reasoning']}")
    print("-" * 70)
