#!/usr/bin/env python3
import json

with open('result.json', 'r') as f:
    results = json.load(f)

total = len(results)
print("=" * 60)
print("预测任务完成总结")
print("=" * 60)
print(f"总完成数: {total} / 10024")
print()

# 统计分数分布
score_dist = {}
for r in results:
    score = r['efficiency_score']
    score_dist[score] = score_dist.get(score, 0) + 1

print("分数分布:")
for score in sorted(score_dist.keys()):
    count = score_dist[score]
    pct = 100 * count / total
    bar = "=" * int(pct / 2)
    print(f"  分数 {score:4.1f}: {count:5d} ({pct:5.2f}%) {bar}")
print()

# 统计理由生成
with_reason = sum(1 for r in results if r.get('reason') and r['reason'].strip())
without_reason = sum(1 for r in results if not r.get('reason') or not r['reason'].strip())

print(f"优化效果统计:")
print(f"  有详细理由: {with_reason:5d} ({100*with_reason/total:5.2f}%)")
print(f"  无理由(快速): {without_reason:5d} ({100*without_reason/total:5.2f}%)")
print()

# 统计需要理由的分子（分数≥7或=1）
need_reason = [r for r in results if r['efficiency_score'] >= 7 or r['efficiency_score'] == 1]
has_reason_when_needed = sum(1 for r in need_reason if r.get('reason', '').strip())

print(f"需要详细理由的分子（分数>=7或=1）:")
print(f"  应该有理由: {len(need_reason):5d}")
print(f"  实际有理由: {has_reason_when_needed:5d}")
if len(need_reason) > 0:
    print(f"  覆盖率: {100*has_reason_when_needed/len(need_reason):.1f}%")
print()

# 高分分子
high_scores = [r for r in results if r['efficiency_score'] >= 7]
print(f"高分分子(>=7分): {len(high_scores)} 个")
if high_scores:
    for s in [10, 9, 8, 7]:
        cnt = sum(1 for r in high_scores if r['efficiency_score'] == s)
        if cnt > 0:
            print(f"  分数{s}: {cnt} 个")
print()

# 低分分子
low_scores = [r for r in results if r['efficiency_score'] == 1]
print(f"极低分分子(=1分): {len(low_scores)} 个")
print()

# 显示Top 10
print("Top 10 高分分子:")
top10 = sorted(results, key=lambda x: x['efficiency_score'], reverse=True)[:10]
for i, r in enumerate(top10, 1):
    smiles = r['SMILES'][:50] + "..." if len(r['SMILES']) > 50 else r['SMILES']
    has_r = "有理由" if r.get('reason', '').strip() else "无理由"
    print(f"  {i:2d}. ID={r['ID']:4d}, Score={r['efficiency_score']}, {has_r}")
    print(f"      {smiles}")
