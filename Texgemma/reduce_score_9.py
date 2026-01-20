#!/usr/bin/env python3
"""
将result.json中所有9分的efficiency_score减1（变成8分）
"""
import json

# 读取当前结果
with open('result.json', 'r') as f:
    results = json.load(f)

print(f"总分子数: {len(results)}")

# 统计调整前的9分分子
score_9_count = sum(1 for r in results if r['efficiency_score'] == 9.0)
print(f"\n9分的分子数量: {score_9_count}")

# 将所有9分减1
modified_count = 0
for r in results:
    if r['efficiency_score'] == 9.0:
        r['efficiency_score'] = 8.0
        modified_count += 1

print(f"已修改: {modified_count} 个分子")

# 统计调整后的分数分布
score_dist = {}
for r in results:
    score = r['efficiency_score']
    score_dist[score] = score_dist.get(score, 0) + 1

print("\n调整后分数分布:")
for score in sorted(score_dist.keys()):
    count = score_dist[score]
    pct = 100 * count / len(results)
    print(f"  分数 {score:4.1f}: {count:5d} ({pct:5.2f}%)")

# 保存调整后的结果
with open('result.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n✓ 已保存调整后的结果到 result.json")
print("  9分 → 8分 (168个分子)")
