"""LANTERN accuracy summary using Within ±2 as the primary metric."""
import pandas as pd
import numpy as np

# Load predictions CSV (update path to match your LANTERN output)
df = pd.read_csv("/mnt/3fs/dots-pretrain/leshu/workspace/LANTERN/LANTERN/results/efficiency_test_qwen/predictions.csv")

preds_cont = df["Prediction"].values
labels = df["Label"].values.astype(int)

# Min-max scale continuous preds to [1, 10]
preds_normalized = (preds_cont - preds_cont.min()) / (preds_cont.max() - preds_cont.min())
preds_scaled = preds_normalized * 9 + 1
preds_rounded = np.round(preds_scaled).astype(int)
preds_rounded = np.clip(preds_rounded, 1, 10)

print("=" * 80)
print("LANTERN accuracy — Within ±2 focus")
print("=" * 80)

extreme_values = [1, 2, 9, 10]
middle_values = [3, 4, 5, 6, 7, 8]

# Extremes
extreme_mask = np.isin(labels, extreme_values)
extreme_labels = labels[extreme_mask]
extreme_preds = preds_rounded[extreme_mask]

extreme_exact = np.mean(extreme_labels == extreme_preds)
extreme_within_1 = np.mean(np.abs(extreme_labels - extreme_preds) <= 1)
extreme_within_2 = np.mean(np.abs(extreme_labels - extreme_preds) <= 2)

print(f"\n[Extremes] (1, 2, 9, 10) — {len(extreme_labels)} samples")
print("-" * 80)
print(f"  Exact (±0):  {extreme_exact*100:6.2f}%  ({int(extreme_exact*len(extreme_labels))}/{len(extreme_labels)})")
print(f"  Within ±1:   {extreme_within_1*100:6.2f}%  ({int(extreme_within_1*len(extreme_labels))}/{len(extreme_labels)})")
print(f"  Within ±2:   {extreme_within_2*100:6.2f}%  ({int(extreme_within_2*len(extreme_labels))}/{len(extreme_labels)})")

# Mid-range
middle_mask = np.isin(labels, middle_values)
middle_labels = labels[middle_mask]
middle_preds = preds_rounded[middle_mask]

middle_exact = np.mean(middle_labels == middle_preds)
middle_within_1 = np.mean(np.abs(middle_labels - middle_preds) <= 1)
middle_within_2 = np.mean(np.abs(middle_labels - middle_preds) <= 2)

print(f"\n[Mid-range] (3, 4, 5, 6, 7, 8) — {len(middle_labels)} samples")
print("-" * 80)
print(f"  Exact (±0):  {middle_exact*100:6.2f}%  ({int(middle_exact*len(middle_labels))}/{len(middle_labels)})")
print(f"  Within ±1:   {middle_within_1*100:6.2f}%  ({int(middle_within_1*len(middle_labels))}/{len(middle_labels)})")
print(f"  Within ±2:   {middle_within_2*100:6.2f}%  ({int(middle_within_2*len(middle_labels))}/{len(middle_labels)})")

# Compare buckets
print("\n[Summary — Within ±2]")
print("=" * 80)
print("Metric                Extreme (1,2,9,10)    Mid (3-8)            Delta")
print("-" * 80)
print(
    f"Count                 {len(extreme_labels):>16}    {len(middle_labels):>19}    "
    f"{len(middle_labels)-len(extreme_labels):>+10}"
)
print(
    f"Within ±2             {extreme_within_2*100:>15.2f}%   {middle_within_2*100:>19.2f}%   "
    f"{(middle_within_2-extreme_within_2)*100:>+9.2f}%"
)
print(
    f"Within ±1             {extreme_within_1*100:>15.2f}%   {middle_within_1*100:>19.2f}%   "
    f"{(middle_within_1-extreme_within_1)*100:>+9.2f}%"
)
print(
    f"Exact                 {extreme_exact*100:>15.2f}%   {middle_exact*100:>19.2f}%   "
    f"{(middle_exact-extreme_exact)*100:>+9.2f}%"
)

improvement_2 = middle_within_2 / extreme_within_2 if extreme_within_2 > 0 else 0
improvement_1 = middle_within_1 / extreme_within_1 if extreme_within_1 > 0 else 0
improvement_exact = middle_exact / extreme_exact if extreme_exact > 0 else 0

print("\n[Mid-range / extreme ratios]")
print("-" * 80)
print(f"  Within ±2: {improvement_2:.2f}x")
print(f"  Within ±1: {improvement_1:.2f}x")
print(f"  Exact:     {improvement_exact:.2f}x")

# Global
total_exact = np.mean(labels == preds_rounded)
total_within_1 = np.mean(np.abs(labels - preds_rounded) <= 1)
total_within_2 = np.mean(np.abs(labels - preds_rounded) <= 2)

print(f"\n[Global] ({len(labels)} rows in CSV)")
print("-" * 80)
print(f"  Exact (±0):  {total_exact*100:6.2f}%  ({int(total_exact*len(labels))}/{len(labels)})")
print(f"  Within ±1:   {total_within_1*100:6.2f}%  ({int(total_within_1*len(labels))}/{len(labels)})")
print(f"  Within ±2:   {total_within_2*100:6.2f}%  ({int(total_within_2*len(labels))}/{len(labels)})")
print("\n" + "=" * 80)
