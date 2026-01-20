#!/usr/bin/env python3
"""
Visualize prediction results comparison:
- LANTERN (on AGILE training data)
- LANTERN (on Haiyu-Qwen30B efficiency test data)
- Comparison plot: True Score vs Predicted Score
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup matplotlib
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")

# Create results directory
os.makedirs('results/visualizations', exist_ok=True)

print("="*80)
print("Efficiency Data: Prediction Result Visualization")
print("="*80)

# ============================================================================
# 1. Load AGILE LANTERN results (training data performance)
# ============================================================================
print("\n1. Loading AGILE training data results...")

agile_csv = '/mnt/3fs/dots-pretrain/leshu/workspace/LANTERN/LANTERN/data/AGILE.csv'
if os.path.exists(agile_csv):
    agile_df = pd.read_csv(agile_csv)
    print(f"   ✓ Loaded {len(agile_df)} AGILE samples")
    print(f"   Label range: [{agile_df['Target'].min():.2f}, {agile_df['Target'].max():.2f}]")
    
    # Try to get LANTERN predictions on AGILE
    lantern_agile_results = '/mnt/3fs/dots-pretrain/leshu/workspace/LANTERN/results/AGILE_predictions.csv'
    if os.path.exists(lantern_agile_results):
        agile_preds_df = pd.read_csv(lantern_agile_results)
        print(f"   ✓ Loaded LANTERN predictions on AGILE: {len(agile_preds_df)} predictions")
    else:
        print(f"   ⚠ LANTERN AGILE predictions not found at {lantern_agile_results}")
        agile_preds_df = None
else:
    print(f"   ✗ AGILE data not found")
    agile_preds_df = None

# ============================================================================
# 2. Load Haiyu-Qwen30B LANTERN results
# ============================================================================
print("\n2. Loading Haiyu-Qwen30B efficiency test data...")

haiyu_csv = '/mnt/3fs/dots-pretrain/leshu/workspace/LANTERN/LANTERN/data/efficiency_test_qwen.csv'
if os.path.exists(haiyu_csv):
    haiyu_df = pd.read_csv(haiyu_csv)
    print(f"   ✓ Loaded {len(haiyu_df)} Haiyu samples")
    print(f"   Label range: [{haiyu_df['Target'].min()}, {haiyu_df['Target'].max()}]")
    
    # Load LANTERN predictions
    haiyu_preds = '/mnt/3fs/dots-pretrain/leshu/workspace/LANTERN/results/efficiency_lantern_results.csv'
    if os.path.exists(haiyu_preds):
        haiyu_preds_df = pd.read_csv(haiyu_preds)
        print(f"   ✓ Loaded LANTERN predictions: {len(haiyu_preds_df)} predictions")
    else:
        print(f"   ✗ LANTERN Haiyu predictions not found")
        haiyu_preds_df = None
else:
    print(f"   ✗ Haiyu data not found")
    haiyu_preds_df = None

# ============================================================================
# 3. Create visualization
# ============================================================================
print("\n3. Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Efficiency Prediction Results Comparison\n(True Score vs Predicted Score)', 
             fontsize=16, fontweight='bold', y=0.995)

# -------- Plot 1: AGILE LANTERN (if available) --------
ax1 = axes[0, 0]
if agile_preds_df is not None:
    true_vals = agile_preds_df['Label'].values
    pred_vals = agile_preds_df['Prediction'].values
    
    ax1.scatter(true_vals, pred_vals, alpha=0.6, s=30, color='steelblue', edgecolors='navy', linewidth=0.5)
    
    # Add diagonal line (perfect prediction)
    min_val = min(true_vals.min(), pred_vals.min())
    max_val = max(true_vals.max(), pred_vals.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Calculate metrics
    mae = np.mean(np.abs(true_vals - pred_vals))
    rmse = np.sqrt(np.mean((true_vals - pred_vals)**2))
    r2 = 1 - np.sum((true_vals - pred_vals)**2) / np.sum((true_vals - true_vals.mean())**2)
    
    ax1.set_xlabel('True Score', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Predicted Score', fontsize=11, fontweight='bold')
    ax1.set_title(f'AGILE Dataset (LANTERN Training Data)\nN={len(true_vals)}, MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}',
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_aspect('equal', adjustable='box')
else:
    ax1.text(0.5, 0.5, 'AGILE Predictions\nNot Available', 
             ha='center', va='center', fontsize=14, color='gray')
    ax1.set_title('AGILE Dataset (LANTERN Training Data)', fontsize=12, fontweight='bold')

# -------- Plot 2: Haiyu LANTERN (continuous predictions) --------
ax2 = axes[0, 1]
if haiyu_preds_df is not None:
    true_vals = haiyu_preds_df['True_Label'].values
    pred_vals = haiyu_preds_df['Predicted'].values
    
    ax2.scatter(true_vals, pred_vals, alpha=0.6, s=30, color='darkorange', edgecolors='darkred', linewidth=0.5)
    
    # Add diagonal line
    min_val = min(true_vals.min(), pred_vals.min())
    max_val = max(true_vals.max(), pred_vals.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Calculate metrics
    mae = np.mean(np.abs(true_vals - pred_vals))
    rmse = np.sqrt(np.mean((true_vals - pred_vals)**2))
    r2 = 1 - np.sum((true_vals - pred_vals)**2) / np.sum((true_vals - true_vals.mean())**2)
    
    ax2.set_xlabel('True Score', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Predicted Score (Continuous)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Haiyu-Qwen30B Efficiency (LANTERN)\nN={len(true_vals)}, MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}',
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xlim(0.5, 10.5)
else:
    ax2.text(0.5, 0.5, 'Haiyu Predictions\nNot Available', 
             ha='center', va='center', fontsize=14, color='gray')
    ax2.set_title('Haiyu-Qwen30B Efficiency (LANTERN)', fontsize=12, fontweight='bold')

# -------- Plot 3: Haiyu LANTERN (rounded predictions) --------
ax3 = axes[1, 0]
if haiyu_preds_df is not None:
    true_vals = haiyu_preds_df['True_Label'].values
    pred_vals_rounded = haiyu_preds_df['Predicted_Rounded'].values
    
    # Add jitter for visibility
    jitter_x = np.random.normal(0, 0.08, size=len(true_vals))
    jitter_y = np.random.normal(0, 0.08, size=len(true_vals))
    ax3.scatter(true_vals + jitter_x, pred_vals_rounded + jitter_y, 
                alpha=0.6, s=30, color='forestgreen', edgecolors='darkgreen', linewidth=0.5)
    
    # Add diagonal line
    ax3.plot([0.5, 10.5], [0.5, 10.5], 'r--', lw=2, label='Perfect Prediction')
    
    # Calculate metrics
    mae = np.mean(np.abs(true_vals - pred_vals_rounded))
    rmse = np.sqrt(np.mean((true_vals - pred_vals_rounded)**2))
    exact_acc = np.mean(true_vals == pred_vals_rounded)
    within_1 = np.mean(np.abs(true_vals - pred_vals_rounded) <= 1)
    within_2 = np.mean(np.abs(true_vals - pred_vals_rounded) <= 2)
    
    ax3.set_xlabel('True Score', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Predicted Score (Rounded [1-10])', fontsize=11, fontweight='bold')
    ax3.set_title(f'Haiyu-Qwen30B Efficiency (LANTERN, Rounded)\nExact Acc={exact_acc*100:.1f}%, ±1={within_1*100:.1f}%, ±2={within_2*100:.1f}%',
                  fontsize=12, fontweight='bold')
    ax3.set_xlim(0.5, 10.5)
    ax3.set_ylim(0.5, 10.5)
    ax3.set_xticks(range(1, 11))
    ax3.set_yticks(range(1, 11))
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
else:
    ax3.text(0.5, 0.5, 'Haiyu Predictions\n(Rounded) Not Available', 
             ha='center', va='center', fontsize=14, color='gray')
    ax3.set_title('Haiyu-Qwen30B Efficiency (LANTERN, Rounded)', fontsize=12, fontweight='bold')

# -------- Plot 4: Distribution Comparison --------
ax4 = axes[1, 1]

# Prepare data for box plots
data_to_plot = []
labels_plot = []

if agile_preds_df is not None:
    data_to_plot.append(agile_preds_df['Label'].values)
    labels_plot.append('AGILE\n(True)')
    data_to_plot.append(agile_preds_df['Prediction'].values)
    labels_plot.append('AGILE\n(Pred)')

if haiyu_preds_df is not None:
    data_to_plot.append(haiyu_preds_df['True_Label'].values)
    labels_plot.append('Haiyu\n(True)')
    data_to_plot.append(haiyu_preds_df['Predicted'].values)
    labels_plot.append('Haiyu\n(Pred)')

if len(data_to_plot) > 0:
    bp = ax4.boxplot(data_to_plot, labels=labels_plot, patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightblue', 'lightsalmon', 'lightsalmon']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
    
    ax4.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax4.set_title('Score Distribution Comparison', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
else:
    ax4.text(0.5, 0.5, 'No Data Available', 
             ha='center', va='center', fontsize=14, color='gray')
    ax4.set_title('Score Distribution Comparison', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('results/visualizations/efficiency_prediction_comparison.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: results/visualizations/efficiency_prediction_comparison.png")

plt.show()

# ============================================================================
# 4. Print summary statistics
# ============================================================================
print("\n" + "="*80)
print("SUMMARY: LANTERN Performance on Different Datasets")
print("="*80)

if agile_preds_df is not None:
    print("\n📊 AGILE Dataset (Training Data):")
    print(f"   • Samples: {len(agile_preds_df)}")
    print(f"   • True Score Range: [{agile_preds_df['Label'].min():.2f}, {agile_preds_df['Label'].max():.2f}]")
    print(f"   • Predicted Range: [{agile_preds_df['Prediction'].min():.2f}, {agile_preds_df['Prediction'].max():.2f}]")
    
    mae = np.mean(np.abs(agile_preds_df['Label'].values - agile_preds_df['Prediction'].values))
    rmse = np.sqrt(np.mean((agile_preds_df['Label'].values - agile_preds_df['Prediction'].values)**2))
    r2 = 1 - np.sum((agile_preds_df['Label'].values - agile_preds_df['Prediction'].values)**2) / \
         np.sum((agile_preds_df['Label'].values - agile_preds_df['Label'].values.mean())**2)
    
    print(f"   • MAE: {mae:.4f}")
    print(f"   • RMSE: {rmse:.4f}")
    print(f"   • R²: {r2:.4f}")

if haiyu_preds_df is not None:
    print("\n📊 Haiyu-Qwen30B Efficiency Test Data:")
    print(f"   • Samples: {len(haiyu_preds_df)}")
    print(f"   • True Score Range: [{haiyu_preds_df['True_Label'].min()}, {haiyu_preds_df['True_Label'].max()}]")
    print(f"   • Predicted Range (continuous): [{haiyu_preds_df['Predicted'].min():.2f}, {haiyu_preds_df['Predicted'].max():.2f}]")
    print(f"   • Predicted Range (rounded): [{haiyu_preds_df['Predicted_Rounded'].min()}, {haiyu_preds_df['Predicted_Rounded'].max()}]")
    
    mae_cont = np.mean(np.abs(haiyu_preds_df['True_Label'].values - haiyu_preds_df['Predicted'].values))
    rmse_cont = np.sqrt(np.mean((haiyu_preds_df['True_Label'].values - haiyu_preds_df['Predicted'].values)**2))
    
    mae_round = np.mean(np.abs(haiyu_preds_df['True_Label'].values - haiyu_preds_df['Predicted_Rounded'].values))
    rmse_round = np.sqrt(np.mean((haiyu_preds_df['True_Label'].values - haiyu_preds_df['Predicted_Rounded'].values)**2))
    
    exact_acc = np.mean(haiyu_preds_df['True_Label'].values == haiyu_preds_df['Predicted_Rounded'].values)
    within_1 = np.mean(np.abs(haiyu_preds_df['True_Label'].values - haiyu_preds_df['Predicted_Rounded'].values) <= 1)
    within_2 = np.mean(np.abs(haiyu_preds_df['True_Label'].values - haiyu_preds_df['Predicted_Rounded'].values) <= 2)
    
    print(f"\n   Continuous Predictions:")
    print(f"   • MAE: {mae_cont:.4f}")
    print(f"   • RMSE: {rmse_cont:.4f}")
    
    print(f"\n   Rounded Predictions [1-10]:")
    print(f"   • MAE: {mae_round:.4f}")
    print(f"   • RMSE: {rmse_round:.4f}")
    print(f"   • Exact Accuracy: {exact_acc*100:.2f}%")
    print(f"   • Within ±1: {within_1*100:.2f}%")
    print(f"   • Within ±2: {within_2*100:.2f}%")

print("\n" + "="*80)
