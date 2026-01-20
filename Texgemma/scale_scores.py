#!/usr/bin/env python3
"""
将result.json中所有的efficiency_score乘以1.5
"""
import json

# 读取原始结果
with open('result.json', 'r') as f:
    results = json.load(f)

print(f"总分子数: {len(results)}")

# 备份原始文件
import shutil
shutil.copy('result.json', 'result_original.json')
print("✓ 已备份原始文件到 result_original.json")

# 统计原始分数分布
original_scores = {}
for r in results:
    score = r['efficiency_score']
    original_scores[score] = original_scores.get(score, 0) + 1

print("\n原始分数分布:")
for score in sorted(original_scores.keys()):
    print(f"  分数 {score}: {original_scores[score]} 个")

# 乘以1.5并限制在10分以内
for r in results:
    original_score = r['efficiency_score']
    new_score = original_score * 1.5
    # 限制最高分为10
    new_score = min(new_score, 10.0)
    r['efficiency_score'] = round(new_score, 1)  # 保留一位小数

# 统计新分数分布
new_scores = {}
for r in results:
    score = r['efficiency_score']
    new_scores[score] = new_scores.get(score, 0) + 1

print("\n调整后分数分布:")
for score in sorted(new_scores.keys()):
    print(f"  分数 {score}: {new_scores[score]} 个")

# 保存调整后的结果
with open('result.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n✓ 已保存调整后的结果到 result.json")
print("\n调整说明:")
print("  - 所有分数 × 1.5")
print("  - 最高分限制为 10.0")
print("  - 原始文件已备份为 result_original.json")
