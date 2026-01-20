"""
使用 RDKit 计算分子描述符（保留一位小数）+ 基于规则的模板化理由生成
这个版本速度快，适合处理大量药物
"""
import pandas as pd
import json
import time
from collections import Counter

print("=" * 70)
print("药物效率分数预测系统 - RDKit + 模板化理由生成版本")
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

print(f"测试模式: {'是' if TEST_MODE else '否'}")
if TEST_MODE:
    print(f"测试数量: {TEST_SIZE}")
print("=" * 70)

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

def score_molecular_weight(mw: float) -> tuple:
    """分子量评分 (理想范围: 160-480)"""
    if 160 <= mw <= 480:
        return 2.0, "理想的分子量"
    elif 480 < mw <= 500:
        return 1.5, "略高的分子量"
    elif 140 <= mw < 160 or 500 < mw <= 520:
        return 1.0, "偏离理想范围的分子量"
    elif 520 < mw <= 600:
        return 0.5, "过高的分子量"
    else:
        return 0.0, "严重超标的分子量" if mw > 600 else "过低的分子量"

def score_lipophilicity(logP: float) -> tuple:
    """脂溶性评分 (理想范围: 0-5)"""
    if 0 <= logP <= 3:
        return 2.0, "适宜的脂溶性"
    elif 3 < logP <= 5:
        return 1.5, "较高的脂溶性"
    elif -0.4 <= logP < 0:
        return 1.0, "偏低的脂溶性"
    elif 5 < logP <= 6:
        return 0.5, "过高的脂溶性"
    else:
        return 0.0, "严重偏离理想范围的脂溶性"

def score_hbond(hbd: int, hba: int) -> tuple:
    """氢键评分"""
    score = 0.0
    reasons = []
    
    if hbd <= 5:
        score += 1.0
        reasons.append("合适的氢键供体数")
    elif hbd <= 7:
        score += 0.5
        reasons.append("较多的氢键供体")
    else:
        reasons.append("过多的氢键供体")
    
    if hba <= 10:
        score += 1.0
        reasons.append("合适的氢键受体数")
    elif hba <= 12:
        score += 0.5
        reasons.append("较多的氢键受体")
    else:
        reasons.append("过多的氢键受体")
    
    return score, "、".join(reasons)

def score_tpsa(tpsa: float) -> tuple:
    """极性表面积评分 (理想: <140)"""
    if tpsa <= 140:
        return 2.0, "理想的极性表面积"
    elif 140 < tpsa <= 160:
        return 1.0, "略高的极性表面积"
    elif 160 < tpsa <= 180:
        return 0.5, "偏高的极性表面积"
    else:
        return 0.0, "过高的极性表面积"

def score_flexibility(rotatable: int) -> tuple:
    """灵活性评分 (理想: <10)"""
    if rotatable <= 10:
        return 2.0, "良好的分子柔性"
    elif 10 < rotatable <= 15:
        return 1.0, "较高的分子柔性"
    elif 15 < rotatable <= 20:
        return 0.5, "过高的分子柔性"
    else:
        return 0.0, "严重过高的分子柔性"

def calculate_efficiency_score_with_reasons(descriptors: dict) -> tuple:
    """基于 Lipinski's Rule of Five 计算效率分数和各项原因"""
    if descriptors is None:
        return 0.0, {}
    
    reasons = {}
    total_score = 0.0
    
    # 分子量
    mw_score, mw_reason = score_molecular_weight(descriptors['molecular_weight'])
    total_score += mw_score
    reasons['molecular_weight'] = mw_reason
    
    # 脂溶性
    logp_score, logp_reason = score_lipophilicity(descriptors['logP'])
    total_score += logp_score
    reasons['lipophilicity'] = logp_reason
    
    # 氢键
    hbond_score, hbond_reason = score_hbond(descriptors['hbd'], descriptors['hba'])
    total_score += hbond_score
    reasons['hydrogen_bonds'] = hbond_reason
    
    # TPSA
    tpsa_score, tpsa_reason = score_tpsa(descriptors['tpsa'])
    total_score += tpsa_score
    reasons['tpsa'] = tpsa_reason
    
    # 柔性
    flex_score, flex_reason = score_flexibility(descriptors['rotatable_bonds'])
    total_score += flex_score
    reasons['flexibility'] = flex_reason
    
    # 加2后保留一位小数
    final_score = round(total_score + 2, 1)
    return min(final_score, 10.0), reasons

def generate_reasoning(descriptors: dict, score: float, reasons: dict) -> str:
    """基于规则生成'由于...所以...'格式的理由"""
    
    # 找出主要问题和优点
    problems = []
    advantages = []
    
    # 分析分子量
    mw = descriptors['molecular_weight']
    if mw > 500:
        problems.append(f"过大的分子量({mw} Da)")
    elif 160 <= mw <= 480:
        advantages.append(f"适宜的分子量({mw} Da)")
    
    # 分析脂溶性
    logp = descriptors['logP']
    if logp > 5:
        problems.append(f"过高的脂溶性(logP={logp})")
    elif 0 <= logp <= 3:
        advantages.append(f"良好的脂溶性(logP={logp})")
    
    # 分析TPSA
    tpsa = descriptors['tpsa']
    if tpsa > 140:
        problems.append(f"较高的极性表面积(TPSA={tpsa}Ų)")
    else:
        advantages.append(f"合适的极性表面积(TPSA={tpsa}Ų)")
    
    # 分析柔性
    rot = descriptors['rotatable_bonds']
    if rot > 10:
        problems.append(f"过多的可旋转键({rot}个)")
    elif rot <= 10:
        advantages.append(f"适度的分子柔性({rot}个可旋转键)")
    
    # 分析氢键
    hbd = descriptors['hbd']
    hba = descriptors['hba']
    if hbd > 5 or hba > 10:
        problems.append(f"氢键参数偏高(供体{hbd}个/受体{hba}个)")
    else:
        advantages.append(f"合适的氢键参数(供体{hbd}个/受体{hba}个)")
    
    # 构建'由于...所以...'格式的reasoning
    if problems:
        # 有主要问题
        problem_str = "、".join(problems[:2])  # 最多列出2个主要问题
        if score >= 7.0:
            result = f"由于这个分子具有{problem_str}等问题，但整体药物特性尚可，所以给予{score}/10的评分。"
        elif score >= 5.0:
            result = f"由于这个分子具有{problem_str}，影响其药物特性，所以给予{score}/10的评分。"
        elif score >= 3.0:
            result = f"由于这个分子具有{problem_str}等明显缺陷，药物特性较差，所以给予{score}/10的评分。"
        else:
            result = f"由于这个分子具有{problem_str}等严重问题，不符合类药五原则，所以给予{score}/10的评分。"
    else:
        # 没有主要问题，主要是优点
        adv_str = "、".join(advantages[:2])  # 最多列出2个主要优点
        result = f"由于这个分子具有{adv_str}等良好的药物特性，符合类药五原则，所以给予{score}/10的评分。"
    
    return result

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
    smiles = row['smile']
    number = row['number']
    
    # 计算描述符
    descriptors = calculate_molecular_descriptors(smiles)
    if descriptors is None:
        print(f"警告: 药物 {number} SMILES 无效")
        continue
    
    # 计算分数和原因
    score, reasons = calculate_efficiency_score_with_reasons(descriptors)
    
    # 生成理由
    reasoning = generate_reasoning(descriptors, score, reasons)
    
    result = {
        'number': number,
        'smiles': smiles,
        'efficiency_score': score,
        'reasoning': reasoning,
        'descriptors': descriptors
    }
    
    results.append(result)
    
    if (idx + 1) % 100 == 0:
        elapsed = time.time() - start_time
        speed = (idx + 1) / elapsed
        eta = (len(df) - idx - 1) / speed
        print(f"进度: {idx + 1}/{len(df)} ({100*(idx+1)/len(df):.1f}%) - "
              f"速度: {speed:.1f} 药物/秒 - ETA: {eta:.1f}秒")

# 按分数排序
results.sort(key=lambda x: x['efficiency_score'], reverse=True)

# 保存结果
output_file = '../data/efficiency_scores_rdkit_with_reasoning.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# 统计
end_time = time.time()
elapsed_time = end_time - start_time

print("\n" + "=" * 70)
print("处理完成！")
print("=" * 70)
print(f"总处理时间: {elapsed_time:.2f} 秒")
print(f"平均处理速度: {len(results)/elapsed_time:.2f} 药物/秒")
print(f"输出文件: {output_file}")

# 分数分布统计
score_counts = Counter(result['efficiency_score'] for result in results)
print("\n分数分布:")
for score in sorted(score_counts.keys(), reverse=True):
    count = score_counts[score]
    percentage = count / len(results) * 100
    print(f"  {score:4.1f} 分: {count:4d} 个药物 ({percentage:5.1f}%)")

# 显示示例
print("\n" + "=" * 70)
print("Top 5 药物示例:")
print("=" * 70)
for i, result in enumerate(results[:5], 1):
    print(f"\n{i}. 药物 {result['number']}: {result['efficiency_score']}/10")
    print(f"   Reasoning: {result['reasoning']}")
    print(f"   SMILES: {result['smiles'][:60]}...")
