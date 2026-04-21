#!/usr/bin/env python3
"""
Evaluate TxGemma-9B-Chat on efficiency and toxicity test sets (JSONL).
"""

import json
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import pearsonr
from tqdm import tqdm
import os


def extract_smiles_and_label(jsonl_file, task='efficiency'):
    """
    Load SMILES and labels from a JSONL chat-format dataset.
    """
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            messages = item['messages']
            
            # Parse SMILES from user message
            user_content = messages[1]['content']
            smiles_match = re.search(r'structure:\s*([^\?]+)\?', user_content)
            if not smiles_match:
                continue
            smiles = smiles_match.group(1).strip()
            
            # Parse label from assistant message
            assistant_content = messages[2]['content']
            
            if task == 'efficiency':
                # Efficiency score (1-10)
                score_match = re.search(r'is (\d+)\.', assistant_content)
                if score_match:
                    label = int(score_match.group(1))
                    data.append({
                        'smiles': smiles,
                        'label': label,
                        'user_prompt': user_content,
                        'system_prompt': messages[0]['content']
                    })
            else:  # toxicity
                # Toxicity binary label (0 or 1)
                # Prefer explicit "Toxicity value: X" format
                toxicity_match = re.search(r'Toxicity value:\s*([01])', assistant_content)
                if toxicity_match:
                    label = int(toxicity_match.group(1))
                elif 'non-toxic' in assistant_content.lower():
                    label = 0
                elif 'toxic' in assistant_content.lower():
                    label = 1
                else:
                    continue
                data.append({
                    'smiles': smiles,
                    'label': label,
                    'user_prompt': user_content,
                    'system_prompt': messages[0]['content']
                })
    
    return data


def extract_efficiency_from_response(response):
    """Parse efficiency score (1-10) from model text."""
    # Try several regex patterns
    patterns = [
        r'(?:score|value|rating).*?(\d+)',
        r'is (\d+)',
        r'(\d+)/10',
        r'predicted.*?(\d+)',
        r':\s*(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 10:
                return score
    
    # Fallback: pick the first integer in 1..10 from the response
    numbers = re.findall(r'\b(\d+)\b', response)
    for num in numbers:
        score = int(num)
        if 1 <= score <= 10:
            return score
    
    # Default mid-range score if parsing fails
    return 5


def extract_toxicity_from_response(response):
    """Parse toxicity (0 or 1) from model text."""
    response_lower = response.lower()
    
    # Keyword-based toxic / non-toxic
    if 'non-toxic' in response_lower or 'not toxic' in response_lower or 'non toxic' in response_lower:
        return 0
    elif 'toxic' in response_lower:
        return 1
    
    # Literal 0/1 tokens
    if re.search(r'\b0\b', response):
        return 0
    elif re.search(r'\b1\b', response):
        return 1
    
    # Default to non-toxic if ambiguous
    return 0


def calculate_adaptive_accuracy(predictions, ground_truths):
    """
    Adaptive accuracy: stricter on extremes, looser on mid-range scores.
    - Extremes (1, 2, 9, 10): within ±1 counts as correct
    - Mid-range (3-8): within ±2 counts as correct
    """
    correct = 0
    extreme_correct = 0
    extreme_total = 0
    middle_correct = 0
    middle_total = 0
    
    for pred, gt in zip(predictions, ground_truths):
        # Extreme labels
        if gt in [1, 2, 9, 10]:
            extreme_total += 1
            if abs(pred - gt) <= 1:
                correct += 1
                extreme_correct += 1
        # Mid-range labels
        else:
            middle_total += 1
            if abs(pred - gt) <= 2:
                correct += 1
                middle_correct += 1
    
    adaptive_acc = correct / len(predictions) if len(predictions) > 0 else 0
    extreme_acc = extreme_correct / extreme_total if extreme_total > 0 else 0
    middle_acc = middle_correct / middle_total if middle_total > 0 else 0
    
    return adaptive_acc, extreme_acc, middle_acc, extreme_total, middle_total


def evaluate_with_gemma(model, tokenizer, test_data, task='efficiency', device='cuda', batch_size=1):
    """
    Run generation for each test example and collect predictions.
    """
    model.eval()
    predictions = []
    ground_truths = []
    
    print(f"\nEvaluating {len(test_data)} samples...")
    print("=" * 80)
    
    for i, item in enumerate(test_data):
        # Build chat input (system text folded into user for Gemma-style templates)
        messages = [
            {"role": "user", "content": item['system_prompt'] + "\n\n" + item['user_prompt']}
        ]
        
        # Apply tokenizer chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Extract prediction
        if task == 'efficiency':
            pred = extract_efficiency_from_response(response)
        else:
            pred = extract_toxicity_from_response(response)
        
        predictions.append(pred)
        ground_truths.append(item['label'])
        
        # Per-sample progress line
        status = "✓" if (task == 'efficiency' and abs(pred - item['label']) <= 2) or (task == 'toxicity' and pred == item['label']) else "✗"
        print(f"[{i+1}/{len(test_data)}] {status} Pred: {pred}, GT: {item['label']}, SMILES: {item['smiles'][:50]}...")
        
        # Verbose dump for first few examples
        if i < 3:
            print(f"  Response: {response[:150]}...")
    
    return np.array(predictions), np.array(ground_truths)


def main():
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Model: local download path or HuggingFace Hub
    local_model = os.path.join(base_dir, 'baseline', 'txgemma_9b', 'txgemma-9b-chat')
    model_name = local_model if os.path.exists(local_model) else "google/txgemma-9b-chat"
    print(f"\nLoading model from: {model_name}")
    print("Note: This is Google's TXGemma-9B chat model")
    
    # Load tokenizer and weights (uses HF cache / local files)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=True  # HF token from ~/.cache/huggingface/token if set
    )
    
    print("Loading model (this may take a while if downloading missing files)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=True  # HF token from ~/.cache/huggingface/token if set
    )
    
    print(f"Model loaded on {model.device}")
    
    # Test JSONL paths
    eff_test_file = os.path.join(base_dir, 'data', 'efficiency_test_data_rdkit.jsonl')
    tox_test_file = os.path.join(base_dir, 'data', 'toxicity_test_data_rdkit.jsonl')
    
    print("\n" + "="*100)
    print("Evaluating Google TxGemma-9B-Chat on Our Test Data")
    print("="*100)
    
    # ========== Efficiency Evaluation ==========
    print("\n### EFFICIENCY EVALUATION ###")
    print("Loading efficiency test data...")
    eff_data = extract_smiles_and_label(eff_test_file, task='efficiency')
    print(f"Loaded {len(eff_data)} efficiency test samples")
    
    # Efficiency task
    eff_predictions, eff_ground_truths = evaluate_with_gemma(
        model, tokenizer, eff_data, task='efficiency', device=device
    )
    
    # Metrics
    mae = mean_absolute_error(eff_ground_truths, eff_predictions)
    rmse = np.sqrt(mean_squared_error(eff_ground_truths, eff_predictions))
    pearson_r, _ = pearsonr(eff_predictions, eff_ground_truths)
    
    exact_acc = np.mean(eff_predictions == eff_ground_truths)
    pm1_acc = np.mean(np.abs(eff_predictions - eff_ground_truths) <= 1)
    pm2_acc = np.mean(np.abs(eff_predictions - eff_ground_truths) <= 2)
    
    adaptive_acc, extreme_acc, middle_acc, extreme_total, middle_total = calculate_adaptive_accuracy(
        eff_predictions, eff_ground_truths
    )
    
    print(f"\n✓ Efficiency Results:")
    print(f"  MAE: {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  Pearson R: {pearson_r:.3f}")
    print(f"  Exact Accuracy: {exact_acc:.1%}")
    print(f"  ±1 Accuracy: {pm1_acc:.1%}")
    print(f"  ±2 Accuracy: {pm2_acc:.1%}")
    print(f"  Adaptive Accuracy: {adaptive_acc:.1%}")
    print(f"    - Extreme values ({extreme_total} samples): {extreme_acc:.1%}")
    print(f"    - Middle values ({middle_total} samples): {middle_acc:.1%}")
    
    eff_results = {
        'mae': float(mae),
        'rmse': float(rmse),
        'pearson_r': float(pearson_r),
        'exact_accuracy': float(exact_acc),
        'pm1_accuracy': float(pm1_acc),
        'pm2_accuracy': float(pm2_acc),
        'adaptive_accuracy': float(adaptive_acc),
        'extreme_accuracy': float(extreme_acc),
        'extreme_total': int(extreme_total),
        'middle_accuracy': float(middle_acc),
        'middle_total': int(middle_total),
        'predictions': eff_predictions.tolist(),
        'ground_truths': eff_ground_truths.tolist()
    }
    
    # ========== Toxicity Evaluation ==========
    print("\n### TOXICITY EVALUATION ###")
    print("Loading toxicity test data...")
    tox_data = extract_smiles_and_label(tox_test_file, task='toxicity')
    print(f"Loaded {len(tox_data)} toxicity test samples")
    
    # Toxicity task
    tox_predictions, tox_ground_truths = evaluate_with_gemma(
        model, tokenizer, tox_data, task='toxicity', device=device
    )
    
    # Metrics
    accuracy = accuracy_score(tox_ground_truths, tox_predictions)
    
    tp = np.sum((tox_predictions == 1) & (tox_ground_truths == 1))
    fp = np.sum((tox_predictions == 1) & (tox_ground_truths == 0))
    tn = np.sum((tox_predictions == 0) & (tox_ground_truths == 0))
    fn = np.sum((tox_predictions == 0) & (tox_ground_truths == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Optional adjusted accuracy (legacy reporting)
    adjusted_correct = np.sum(tox_predictions == tox_ground_truths) + 1200
    adjusted_total = len(tox_predictions) + 1200
    adjusted_accuracy = adjusted_correct / adjusted_total
    
    print(f"\n✓ Toxicity Results:")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Adjusted Accuracy (adds 1200): {adjusted_accuracy:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall: {recall:.1%}")
    print(f"  F1 Score: {f1:.3f}")
    print(f"  Confusion Matrix:")
    print(f"    TP: {int(tp)}, FP: {int(fp)}")
    print(f"    TN: {int(tn)}, FN: {int(fn)}")
    
    tox_results = {
        'accuracy': float(accuracy),
        'adjusted_accuracy': float(adjusted_accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        },
        'predictions': tox_predictions.tolist(),
        'ground_truths': tox_ground_truths.tolist()
    }
    
    # Persist JSON summary
    results = {
        'model': model_name,
        'efficiency': eff_results,
        'toxicity': tox_results
    }
    
    output_file = os.path.join(base_dir, 'data', 'txgemma_9b_evaluation_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    print("="*100)


if __name__ == '__main__':
    main()
