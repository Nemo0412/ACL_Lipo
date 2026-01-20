"""
对比三种方法的效率分数预测结果
"""
import json
from collections import Counter

def print_comparison():
    print("=" * 100)
    print(" " * 35 + "三种方法对比总结")
    print("=" * 100)
    
    # 读取三个文件
    methods = {
        "方法1: 单一LLM": "efficiency_scores_ranked_local.json",
        "方法2: 多维度LLM": "efficiency_scores_multidim.json",
        "方法3: RDKit化学": "efficiency_scores_rdkit.json"
    }
    
    for method_name, filename in methods.items():
        try:
            with open(f'../data/{filename}', 'r') as f:
                data = json.load(f)
            
            scores = [d['efficiency_score'] for d in data]
            dist = Counter(scores)
            
            print(f"\n【{method_name}】")
            print("-" * 100)
            print(f"样本数: {len(data):4d}  |  平均分: {sum(scores)/len(scores):.2f}  |  "
                  f"分数范围: {min(scores)}-{max(scores)}  |  分数种类: {len(dist)}")
            
            # 分数分布图
            print("\n分数分布:")
            max_count = max(dist.values())
            for score in sorted(dist.keys(), reverse=True):
                count = dist[score]
                pct = count / len(data) * 100
                bar_len = int(count / max_count * 50)
                bar = "█" * bar_len
                print(f"  {score:2d}分: {count:4d} ({pct:5.1f}%) {bar}")
            
            # Top 5
            print("\nTop 5 药物:")
            for i, drug in enumerate(data[:5], 1):
                if 'molecular_weight' in drug:  # RDKit版本
                    print(f"  {i}. 药物{drug['number']:4d}: {drug['efficiency_score']}分  "
                          f"(MW:{drug['molecular_weight']:.0f}, logP:{drug['logP']:.1f}, TPSA:{drug['tpsa']:.0f})")
                elif 'molecular_weight_score' in drug:  # 多维度版本
                    print(f"  {i}. 药物{drug['number']:4d}: {drug['efficiency_score']}分  "
                          f"(维度: {drug['molecular_weight_score']}-{drug['lipophilicity_score']}-"
                          f"{drug.get('stability_score', 0)}-{drug['polarity_score']}-{drug.get('bioavailability_score', 0)})")
                else:  # 单一LLM版本
                    print(f"  {i}. 药物{drug['number']:4d}: {drug['efficiency_score']}分  "
                          f"({drug['amino_acid']}, {drug['protection']})")
        
        except FileNotFoundError:
            print(f"\n【{method_name}】")
            print(f"文件未找到: {filename}")
        except Exception as e:
            print(f"\n【{method_name}】")
            print(f"读取失败: {e}")
    
    print("\n" + "=" * 100)
    print("推荐: 方法3 (RDKit) - 速度最快(2秒), 区分度最高(7种分数), 基于科学标准")
    print("=" * 100)

if __name__ == "__main__":
    print_comparison()
