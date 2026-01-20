#!/usr/bin/env python3
"""
Use LANTERN pretrained model to evaluate Haiyu-Qwen30B test data
"""
import os
import sys
sys.path.insert(0, '/mnt/3fs/dots-pretrain/leshu/workspace/LANTERN/LANTERN')

import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report
import pickle

# Load our test data
def load_fingerprints(data_name):
    """Load circular and expert fingerprints"""
    fp_dir = f'/mnt/3fs/dots-pretrain/leshu/workspace/LANTERN/LANTERN/data/fingerprints/{data_name}'
    
    with open(f'{fp_dir}/circular.pkl', 'rb') as f:
        circular_fps = pickle.load(f)
    
    with open(f'{fp_dir}/expert.pkl', 'rb') as f:
        expert_fps = pickle.load(f)
    
    return circular_fps, expert_fps

def load_labels(csv_path, label_col='Target'):
    """Load labels from CSV"""
    df = pd.read_csv(csv_path)
    return df['SMILES'].tolist(), df[label_col].tolist()

def prepare_features(smiles_list, circular_fps, expert_fps):
    """Prepare combined features, truncating expert to 210 dims"""
    features = []
    for smiles in smiles_list:
        circular = circular_fps[smiles]  # 2048 dims
        expert = expert_fps[smiles][:210]  # Truncate to 210 dims (was 217)
        combined = np.concatenate([circular, expert])
        features.append(combined)
    return np.array(features)

def evaluate_efficiency():
    """Evaluate LANTERN on efficiency test data"""
    print("\n" + "="*80)
    print("Evaluating LANTERN on Efficiency Test Data")
    print("="*80)
    
    # Load data
    smiles_list, labels = load_labels('/mnt/3fs/dots-pretrain/leshu/workspace/LANTERN/LANTERN/data/efficiency_test_qwen.csv')
    circular_fps, expert_fps = load_fingerprints('efficiency_test_qwen')
    
    print(f"Loaded {len(smiles_list)} samples")
    print(f"Label distribution: {pd.Series(labels).value_counts().sort_index().to_dict()}")
    
    # Prepare features
    features = prepare_features(smiles_list, circular_fps, expert_fps)
    print(f"Feature shape: {features.shape}")  # Should be (600, 2258)
    
    # Load scaler
    scaler = joblib.load('/mnt/3fs/dots-pretrain/leshu/workspace/LANTERN/LANTERN/checkpoints/circular-expert-model-MLP_scaler.pkl')
    print(f"Scaler expects: {scaler.n_features_in_} features")
    
    # Scale features (add dummy label column first)
    dummy_labels = np.zeros((len(features), 1))
    tmp = np.hstack((features, dummy_labels))
    print(f"Combined shape before scaling: {tmp.shape}")
    
    # Scale
    scaled = scaler.transform(tmp)
    scaled_features = scaled[:, :-1]  # Remove dummy label column
    
    print(f"Scaled features shape: {scaled_features.shape}")
    
    # Load model
    from models.neural_models import FeedforwardRegressor
    model = FeedforwardRegressor(input_count=scaled_features.shape[1], output_count=1)
    model.load_state_dict(torch.load('/mnt/3fs/dots-pretrain/leshu/workspace/LANTERN/LANTERN/checkpoints/circular-expert-model-MLP.pth', map_location='cpu'))
    model.eval()
    
    # Predict
    with torch.no_grad():
        scaled_features_tensor = torch.FloatTensor(scaled_features)
        predictions = model(scaled_features_tensor).numpy().flatten()
    
    # Inverse transform predictions - use original features + predictions
    # This matches LANTERN's original inference logic
    pred_with_features = np.hstack((features, predictions.reshape(-1, 1)))
    unscaled = scaler.inverse_transform(pred_with_features)
    final_predictions = unscaled[:, -1]
    
    print(f"\nPrediction range before rounding: [{final_predictions.min():.2f}, {final_predictions.max():.2f}]")
    print(f"Prediction mean: {final_predictions.mean():.2f}, std: {final_predictions.std():.2f}")
    
    # Round to nearest integer and clip to [1, 10] range (efficiency scores are 1-10)
    rounded_predictions = np.round(final_predictions).astype(int)
    rounded_predictions = np.clip(rounded_predictions, 1, 10)
    
    print(f"After rounding and clipping to [1,10]: unique values = {np.unique(rounded_predictions)}")
    
    # Calculate metrics
    mae = mean_absolute_error(labels, final_predictions)
    rmse = np.sqrt(mean_squared_error(labels, final_predictions))
    mae_rounded = mean_absolute_error(labels, rounded_predictions)
    rmse_rounded = np.sqrt(mean_squared_error(labels, rounded_predictions))
    
    # Accuracy metrics (using rounded predictions)
    exact_acc = accuracy_score(labels, rounded_predictions)
    within_1 = np.mean(np.abs(np.array(labels) - rounded_predictions) <= 1)
    within_2 = np.mean(np.abs(np.array(labels) - rounded_predictions) <= 2)
    
    print("\n" + "="*80)
    print("LANTERN Efficiency Results:")
    print("="*80)
    print(f"Continuous Predictions:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"\nRounded Predictions (1-10):")
    print(f"  MAE: {mae_rounded:.4f}")
    print(f"  RMSE: {rmse_rounded:.4f}")
    print(f"  Exact Accuracy: {exact_acc:.4f} ({exact_acc*100:.2f}%)")
    print(f"  Within ±1: {within_1:.4f} ({within_1*100:.2f}%)")
    print(f"  Within ±2: {within_2:.4f} ({within_2*100:.2f}%)")
    
    # Save results
    os.makedirs('/mnt/3fs/dots-pretrain/leshu/workspace/LANTERN/results', exist_ok=True)
    results_df = pd.DataFrame({
        'SMILES': smiles_list,
        'True_Label': labels,
        'Predicted': final_predictions,
        'Predicted_Rounded': rounded_predictions,
        'Error': np.abs(np.array(labels) - rounded_predictions)
    })
    results_df.to_csv('/mnt/3fs/dots-pretrain/leshu/workspace/LANTERN/results/efficiency_lantern_results.csv', index=False)
    print(f"\nResults saved to: results/efficiency_lantern_results.csv")
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAE_Rounded': mae_rounded,
        'RMSE_Rounded': rmse_rounded,
        'Exact_Accuracy': exact_acc,
        'Within_1': within_1,
        'Within_2': within_2
    }

def evaluate_toxicity():
    """Evaluate LANTERN on toxicity test data"""
    print("\n" + "="*80)
    print("Evaluating LANTERN on Toxicity Test Data")
    print("="*80)
    
    # Load data
    smiles_list, labels = load_labels('/mnt/3fs/dots-pretrain/leshu/workspace/LANTERN/LANTERN/data/toxicity_test_qwen.csv')
    circular_fps, expert_fps = load_fingerprints('toxicity_test_qwen')
    
    print(f"Loaded {len(smiles_list)} samples")
    print(f"Label distribution: {pd.Series(labels).value_counts().sort_index().to_dict()}")
    
    # Prepare features
    features = prepare_features(smiles_list, circular_fps, expert_fps)
    print(f"Feature shape: {features.shape}")
    
    # Load scaler
    scaler = joblib.load('/mnt/3fs/dots-pretrain/leshu/workspace/LANTERN/LANTERN/checkpoints/circular-expert-model-MLP_scaler.pkl')
    
    # Add dummy label column for scaling
    labels_arr = np.array(labels).reshape(-1, 1).astype(float)
    tmp = np.hstack((features, labels_arr))
    
    # Scale
    scaled = scaler.transform(tmp)
    scaled_features = scaled[:, :-1]
    
    # Load model
    from models.neural_models import FeedforwardRegressor
    model = FeedforwardRegressor(input_count=scaled_features.shape[1], output_count=1)
    model.load_state_dict(torch.load('/mnt/3fs/dots-pretrain/leshu/workspace/LANTERN/LANTERN/checkpoints/circular-expert-model-MLP.pth', map_location='cpu'))
    model.eval()
    
    # Predict
    with torch.no_grad():
        scaled_features_tensor = torch.FloatTensor(scaled_features)
        predictions = model(scaled_features_tensor).numpy().flatten()
    
    # Inverse transform predictions
    pred_with_features = np.hstack((features, predictions.reshape(-1, 1)))
    unscaled = scaler.inverse_transform(pred_with_features)
    final_predictions = unscaled[:, -1]
    
    print(f"\nPrediction range: [{final_predictions.min():.2f}, {final_predictions.max():.2f}]")
    print(f"Prediction mean: {final_predictions.mean():.2f}, std: {final_predictions.std():.2f}")
    
    # Convert to binary classification (threshold at 0.5)
    # But first, we need to map the continuous predictions to [0,1] range
    # Since AGILE range is roughly [-2, 16], we'll use a threshold based on the median
    # Or simply use 0.5 as threshold after checking the prediction distribution
    binary_predictions = (final_predictions > 0.5).astype(int)
    
    print(f"Binary predictions distribution: 0={np.sum(binary_predictions==0)}, 1={np.sum(binary_predictions==1)}")
    
    # Calculate metrics
    accuracy = accuracy_score(labels, binary_predictions)
    
    print("\n" + "="*80)
    print("LANTERN Toxicity Results:")
    print("="*80)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(labels, binary_predictions, target_names=['Non-toxic', 'Toxic']))
    
    # Save results
    results_df = pd.DataFrame({
        'SMILES': smiles_list,
        'True_Label': labels,
        'Predicted_Score': final_predictions,
        'Predicted_Binary': binary_predictions,
        'Correct': (np.array(labels) == binary_predictions).astype(int)
    })
    results_df.to_csv('/mnt/3fs/dots-pretrain/leshu/workspace/LANTERN/results/toxicity_lantern_results.csv', index=False)
    print(f"\nResults saved to: results/toxicity_lantern_results.csv")
    
    return {
        'Accuracy': accuracy,
        'Predictions': final_predictions,
        'Binary_Predictions': binary_predictions
    }

if __name__ == '__main__':
    # Evaluate efficiency
    eff_metrics = evaluate_efficiency()
    
    # Evaluate toxicity
    tox_metrics = evaluate_toxicity()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: LANTERN Performance on Haiyu-Qwen30B Test Data")
    print("="*80)
    print("\nEfficiency Task (600 samples):")
    print(f"  Continuous Predictions:")
    print(f"    - MAE: {eff_metrics['MAE']:.4f}")
    print(f"    - RMSE: {eff_metrics['RMSE']:.4f}")
    print(f"  Rounded to Integer [1-10]:")
    print(f"    - MAE: {eff_metrics['MAE_Rounded']:.4f}")
    print(f"    - RMSE: {eff_metrics['RMSE_Rounded']:.4f}")
    print(f"    - Exact Accuracy: {eff_metrics['Exact_Accuracy']*100:.2f}%")
    print(f"    - Within ±1: {eff_metrics['Within_1']*100:.2f}%")
    print(f"    - Within ±2: {eff_metrics['Within_2']*100:.2f}%")
    
    print("\nToxicity Task (200 samples):")
    print(f"  - Accuracy: {tox_metrics['Accuracy']*100:.2f}%")
    print("\n" + "="*80)
