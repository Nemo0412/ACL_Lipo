#!/usr/bin/env python3
"""
Analyze LANTERN efficiency predictions by label range (extreme vs middle scores).
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

# Load predictions CSV (update path via run_evaluation pipeline output)
df = pd.read_csv("/mnt/3fs/dots-pretrain/leshu/workspace/LANTERN/LANTERN/results/efficiency_test_qwen/predictions.csv")

preds_cont = df["Prediction"].values
labels = df["Label"].values.astype(int)

# Min-max scale continuous preds to [1, 10]
preds_normalized = (preds_cont - preds_cont.min()) / (preds_cont.max() - preds_cont.min())
preds_scaled = preds_normalized * 9 + 1
preds_rounded = np.round(preds_scaled).astype(int)
preds_rounded = np.clip(preds_rounded, 1, 10)

print("=" * 80)
print("LANTERN efficiency accuracy — extremes vs mid-range labels")
print("=" * 80)

extreme_values = [1, 2, 9, 10]
middle_values = [3, 4, 5, 6, 7, 8]

# 1) Extreme labels
print("\n[Extreme labels] (1, 2, 9, 10)")
print("-" * 80)

extreme_mask = np.isin(labels, extreme_values)
extreme_labels = labels[extreme_mask]
extreme_preds = preds_rounded[extreme_mask]

if len(extreme_labels) > 0:
    extreme_acc = accuracy_score(extreme_labels, extreme_preds)
    extreme_mae = mean_absolute_error(extreme_labels, extreme_preds)
    extreme_rmse = np.sqrt(mean_squared_error(extreme_labels, extreme_preds))
    extreme_within_1 = np.mean(np.abs(extreme_labels - extreme_preds) <= 1)
    extreme_within_2 = np.mean(np.abs(extreme_labels - extreme_preds) <= 2)

    print(f"Count: {len(extreme_labels)}")
    print(f"Label histogram: {dict(zip(*np.unique(extreme_labels, return_counts=True)))}")
    print("\nMetrics:")
    print(f"  MAE: {extreme_mae:.4f}")
    print(f"  RMSE: {extreme_rmse:.4f}")
    print(f"  Exact Accuracy: {extreme_acc:.4f} ({extreme_acc*100:.2f}%)")
    print(f"  Within ±1: {extreme_within_1:.4f} ({extreme_within_1*100:.2f}%)")
    print(f"  Within ±2: {extreme_within_2:.4f} ({extreme_within_2*100:.2f}%)")

    print("\nPer-label breakdown:")
    for val in extreme_values:
        val_mask = labels == val
        if np.sum(val_mask) > 0:
            val_labels = labels[val_mask]
            val_preds = preds_rounded[val_mask]
            val_acc = accuracy_score(val_labels, val_preds)
            val_mae = mean_absolute_error(val_labels, val_preds)
            val_within_1 = np.mean(np.abs(val_labels - val_preds) <= 1)
            print(
                f"  label={val}: n={np.sum(val_mask)}, acc={val_acc*100:.2f}%, "
                f"MAE={val_mae:.3f}, within±1={val_within_1*100:.2f}%"
            )
else:
    print("No extreme-label samples.")

# 2) Mid-range labels
print("\n" + "=" * 80)
print("[Mid-range labels] (3, 4, 5, 6, 7, 8)")
print("-" * 80)

middle_mask = np.isin(labels, middle_values)
middle_labels = labels[middle_mask]
middle_preds = preds_rounded[middle_mask]

if len(middle_labels) > 0:
    middle_acc = accuracy_score(middle_labels, middle_preds)
    middle_mae = mean_absolute_error(middle_labels, middle_preds)
    middle_rmse = np.sqrt(mean_squared_error(middle_labels, middle_preds))
    middle_within_1 = np.mean(np.abs(middle_labels - middle_preds) <= 1)
    middle_within_2 = np.mean(np.abs(middle_labels - middle_preds) <= 2)

    print(f"Count: {len(middle_labels)}")
    print(f"Label histogram: {dict(zip(*np.unique(middle_labels, return_counts=True)))}")
    print("\nMetrics:")
    print(f"  MAE: {middle_mae:.4f}")
    print(f"  RMSE: {middle_rmse:.4f}")
    print(f"  Exact Accuracy: {middle_acc:.4f} ({middle_acc*100:.2f}%)")
    print(f"  Within ±1: {middle_within_1:.4f} ({middle_within_1*100:.2f}%)")
    print(f"  Within ±2: {middle_within_2:.4f} ({middle_within_2*100:.2f}%)")

    print("\nPer-label breakdown:")
    for val in middle_values:
        val_mask = labels == val
        if np.sum(val_mask) > 0:
            val_labels = labels[val_mask]
            val_preds = preds_rounded[val_mask]
            val_acc = accuracy_score(val_labels, val_preds)
            val_mae = mean_absolute_error(val_labels, val_preds)
            val_within_1 = np.mean(np.abs(val_labels - val_preds) <= 1)
            print(
                f"  label={val}: n={np.sum(val_mask)}, acc={val_acc*100:.2f}%, "
                f"MAE={val_mae:.3f}, within±1={val_within_1*100:.2f}%"
            )
else:
    print("No mid-range samples.")

# 3) Side-by-side summary
print("\n" + "=" * 80)
print("[Summary]")
print("=" * 80)

if len(extreme_labels) > 0 and len(middle_labels) > 0:
    print(f"\n{'Metric':<20} {'Extreme (1,2,9,10)':<20} {'Middle (3-8)':<20} {'Delta':<20}")
    print("-" * 80)
    print(f"{'Count':<20} {len(extreme_labels):<20} {len(middle_labels):<20} {len(middle_labels)-len(extreme_labels):<20}")
    print(
        f"{'Exact Accuracy':<20} {f'{extreme_acc*100:.2f}%':<20} {f'{middle_acc*100:.2f}%':<20} "
        f"{f'{(middle_acc-extreme_acc)*100:+.2f}%':<20}"
    )
    print(
        f"{'Within ±1':<20} {f'{extreme_within_1*100:.2f}%':<20} {f'{middle_within_1*100:.2f}%':<20} "
        f"{f'{(middle_within_1-extreme_within_1)*100:+.2f}%':<20}"
    )
    print(
        f"{'Within ±2':<20} {f'{extreme_within_2*100:.2f}%':<20} {f'{middle_within_2*100:.2f}%':<20} "
        f"{f'{(middle_within_2-extreme_within_2)*100:+.2f}%':<20}"
    )
    print(f"{'MAE':<20} {f'{extreme_mae:.4f}':<20} {f'{middle_mae:.4f}':<20} {f'{middle_mae-extreme_mae:+.4f}':<20}")
    print(f"{'RMSE':<20} {f'{extreme_rmse:.4f}':<20} {f'{middle_rmse:.4f}':<20} {f'{middle_rmse-extreme_rmse:+.4f}':<20}")

# 4) Global
print("\n" + "=" * 80)
print("[Global] (all rows in CSV)")
print("-" * 80)

total_acc = accuracy_score(labels, preds_rounded)
total_mae = mean_absolute_error(labels, preds_rounded)
total_rmse = np.sqrt(mean_squared_error(labels, preds_rounded))
total_within_1 = np.mean(np.abs(labels - preds_rounded) <= 1)
total_within_2 = np.mean(np.abs(labels - preds_rounded) <= 2)

print(f"Count: {len(labels)}")
print(f"Label histogram: {dict(zip(*np.unique(labels, return_counts=True)))}")
print("\nMetrics:")
print(f"  MAE: {total_mae:.4f}")
print(f"  RMSE: {total_rmse:.4f}")
print(f"  Exact Accuracy: {total_acc:.4f} ({total_acc*100:.2f}%)")
print(f"  Within ±1: {total_within_1:.4f} ({total_within_1*100:.2f}%)")
print(f"  Within ±2: {total_within_2:.4f} ({total_within_2*100:.2f}%)")

# 5) Pred vs label histograms
print("\n" + "=" * 80)
print("[Distribution check]")
print("-" * 80)
print(f"Prediction histogram: {dict(zip(*np.unique(preds_rounded, return_counts=True)))}")
print(f"Label histogram: {dict(zip(*np.unique(labels, return_counts=True)))}")

# 6) Top misprediction pairs
print("\n" + "=" * 80)
print("[Confusion-style errors] (top 10 off-diagonal pairs)")
print("-" * 80)
confusion = defaultdict(lambda: defaultdict(int))
for true_val, pred_val in zip(labels, preds_rounded):
    confusion[true_val][pred_val] += 1

print("\nTrue label → predicted label (most frequent mistakes)")
errors = []
for true_val in sorted(confusion.keys()):
    for pred_val in sorted(confusion[true_val].keys()):
        count = confusion[true_val][pred_val]
        if true_val != pred_val and count > 0:
            errors.append((count, true_val, pred_val))

errors.sort(reverse=True)
for count, true_val, pred_val in errors[:10]:
    print(f"  true={true_val} → pred={pred_val}: {count} times")

print("\n" + "=" * 80)
