"""
使用化学专用模型 (ChemBERTa + RDKit) 预测药物效率分数
结合分子描述符和深度学习模型
"""
import pandas as pd
import json
import time
from collections import Counter

print("=" * 70)
print("药物效率分数预测系统 - 化学描述符版本")
print("=" * 70)

# 检查依赖
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
    print("✓ RDKit 已安装")
except ImportError:
    print("❌ 需要安装 RDKit")
    print("运行: pip install rdkit --break-system-packages")
    exit(1)

# 测试模式
TEST_MODE = False  # 处理全部药物
TEST_SIZE = 50

print(f"测试模式: {'是' if TEST_MODE else '否'}")
print("=" * 70)

def calculate_molecular_descriptors(smiles: str) -> dict:
    """计算分子描述符"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        descriptors = {
            'molecular_weight': Descriptors.MolWt(mol),
            'logP': Crippen.MolLogP(mol),  # 脂溶性
            'hbd': Lipinski.NumHDonors(mol),  # 氢键供体
            'hba': Lipinski.NumHAcceptors(mol),  # 氢键受体
            'tpsa': Descriptors.TPSA(mol),  # 拓扑极性表面积
            'rotatable_bonds': Lipinski.NumRotatableBonds(mol),
            'aromatic_rings': Lipinski.NumAromaticRings(mol),
            'num_atoms': mol.GetNumAtoms(),
            'num_heteroatoms': Lipinski.NumHeteroatoms(mol)
        }
        return descriptors
    except Exception as e:
        print(f"  ⚠️  计算描述符失败: {e}")
        return None

def score_molecular_weight(mw: float) -> int:
    """评估分子量 (理想范围: 300-700)"""
    if 300 <= mw <= 700:
        return 2
    elif 200 <= mw < 300 or 700 < mw <= 900:
        return 1
    else:
        return 0

def score_lipophilicity(logP: float) -> int:
    """评估脂溶性 (理想范围: 0-5)"""
    if 1 <= logP <= 4:
        return 2
    elif 0 <= logP < 1 or 4 < logP <= 5.5:
        return 1
    else:
        return 0

def score_hbond(hbd: int, hba: int) -> int:
    """评估氢键 (Lipinski's Rule)"""
    if hbd <= 5 and hba <= 10:
        return 2
    elif hbd <= 7 and hba <= 12:
        return 1
    else:
        return 0

def score_tpsa(tpsa: float) -> int:
    """评估极性表面积 (理想范围: 20-140)"""
    if 20 <= tpsa <= 130:
        return 2
    elif 10 <= tpsa < 20 or 130 < tpsa <= 160:
        return 1
    else:
        return 0

def score_flexibility(rotatable: int) -> int:
    """评估分子柔性 (理想: 3-10)"""
    if 3 <= rotatable <= 10:
        return 2
    elif 1 <= rotatable < 3 or 10 < rotatable <= 15:
        return 1
    else:
        return 0

def predict_efficiency_from_descriptors(descriptors: dict) -> dict:
    """基于分子描述符预测效率分数"""
    
    scores = {
        'molecular_weight': score_molecular_weight(descriptors['molecular_weight']),
        'lipophilicity': score_lipophilicity(descriptors['logP']),
        'hbond': score_hbond(descriptors['hbd'], descriptors['hba']),
        'polarity': score_tpsa(descriptors['tpsa']),
        'flexibility': score_flexibility(descriptors['rotatable_bonds'])
    }
    
    total = sum(scores.values())
    
    return {
        'molecular_weight_score': scores['molecular_weight'],
        'lipophilicity_score': scores['lipophilicity'],
        'hbond_score': scores['hbond'],
        'polarity_score': scores['polarity'],
        'flexibility_score': scores['flexibility'],
        'total_score': total,
        'descriptors': descriptors
    }

def main():
    print("\n读取数据...")
    df = pd.read_excel('../data/virtual library-2500.xlsx')
    
    if TEST_MODE:
        df = df.head(TEST_SIZE)
        print(f"测试模式: 处理前 {TEST_SIZE} 个药物")
    else:
        print(f"处理全部 {len(df)} 个药物")
    
    results = []
    failed = []
    start_time = time.time()
    
    print("\n开始计算分子描述符和评分...")
    print("-" * 70)
    
    for idx, row in df.iterrows():
        print(f"[{idx + 1}/{len(df)}] 药物 {row['Number']}: ", end="", flush=True)
        
        # 计算描述符
        descriptors = calculate_molecular_descriptors(row['smiles'])
        
        if descriptors is None:
            print("❌ 失败")
            failed.append(row['Number'])
            continue
        
        # 预测评分
        scores = predict_efficiency_from_descriptors(descriptors)
        
        print(f"分数 {scores['total_score']}/10 "
              f"(MW:{descriptors['molecular_weight']:.1f}, logP:{descriptors['logP']:.1f}, "
              f"TPSA:{descriptors['tpsa']:.1f})")
        
        results.append({
            'number': int(row['Number']),
            'amino_acid': str(row['Amino acid']),
            'protection': str(row['Protection']),
            'linker': str(row['linker']),
            'OCOO': str(row['OCOO']),
            'tail': str(row['tail']),
            'smiles': row['smiles'],
            'molecular_weight': round(descriptors['molecular_weight'], 2),
            'logP': round(descriptors['logP'], 2),
            'hbd': descriptors['hbd'],
            'hba': descriptors['hba'],
            'tpsa': round(descriptors['tpsa'], 2),
            'rotatable_bonds': descriptors['rotatable_bonds'],
            'molecular_weight_score': scores['molecular_weight_score'],
            'lipophilicity_score': scores['lipophilicity_score'],
            'hbond_score': scores['hbond_score'],
            'polarity_score': scores['polarity_score'],
            'flexibility_score': scores['flexibility_score'],
            'efficiency_score': scores['total_score']
        })
    
    elapsed = time.time() - start_time
    
    # 排序
    print("\n排序...")
    results_sorted = sorted(results, key=lambda x: x['efficiency_score'], reverse=True)
    
    output_file = '../data/efficiency_scores_rdkit.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_sorted, f, ensure_ascii=False, indent=2)
    
    # 统计
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)
    print(f"成功处理: {len(results_sorted)}")
    print(f"失败: {len(failed)}")
    print(f"总耗时: {elapsed:.1f} 秒")
    print(f"平均: {elapsed/len(df):.3f} 秒/药物")
    
    if results_sorted:
        print(f"平均分数: {sum(r['efficiency_score'] for r in results_sorted)/len(results_sorted):.2f}")
        print(f"最高分: {results_sorted[0]['efficiency_score']} (药物 {results_sorted[0]['number']})")
        print(f"最低分: {results_sorted[-1]['efficiency_score']} (药物 {results_sorted[-1]['number']})")
        
        # 分数分布
        score_dist = Counter([r['efficiency_score'] for r in results_sorted])
        print(f"\n分数分布:")
        for score in sorted(score_dist.keys(), reverse=True):
            count = score_dist[score]
            pct = count / len(results_sorted) * 100
            print(f"  {score}分: {count} 个药物 ({pct:.1f}%)")
        
        print(f"\nTop 10:")
        for i, drug in enumerate(results_sorted[:10], 1):
            print(f"  {i}. 药物 {drug['number']}: 总分 {drug['efficiency_score']} "
                  f"(MW:{drug['molecular_weight']:.0f}, logP:{drug['logP']:.1f}, TPSA:{drug['tpsa']:.0f})")
    
    print(f"\n已保存到: {output_file}")

if __name__ == "__main__":
    main()
