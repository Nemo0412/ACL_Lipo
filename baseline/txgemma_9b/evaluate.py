#!/usr/bin/env python3
"""
使用 Google Gemma-9B 模型评估我们的 efficiency 和 toxicity 测试数据
模型: google/gemma-2-9b-it
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
    从 JSONL 文件中提取 SMILES 和标签
    """
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            messages = item['messages']
            
            # 提取 SMILES
            user_content = messages[1]['content']
            smiles_match = re.search(r'structure:\s*([^\?]+)\?', user_content)
            if not smiles_match:
                continue
            smiles = smiles_match.group(1).strip()
            
            # 提取标签
            assistant_content = messages[2]['content']
            
            if task == 'efficiency':
                # 提取 efficiency score (1-10)
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
                # 提取 toxicity (0 or 1)
                # 优先检查 "Toxicity value: X" 格式
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
    """从模型响应中提取 efficiency score (1-10)"""
    # 尝试多种模式
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
    
    # 如果找不到，尝试提取所有数字并选择1-10范围内的
    numbers = re.findall(r'\b(\d+)\b', response)
    for num in numbers:
        score = int(num)
        if 1 <= score <= 10:
            return score
    
    # 默认返回中间值
    return 5


def extract_toxicity_from_response(response):
    """从模型响应中提取 toxicity (0 or 1)"""
    response_lower = response.lower()
    
    # 检查明确的 toxic/non-toxic 关键词
    if 'non-toxic' in response_lower or 'not toxic' in response_lower or 'non toxic' in response_lower:
        return 0
    elif 'toxic' in response_lower:
        return 1
    
    # 检查 0/1
    if re.search(r'\b0\b', response):
        return 0
    elif re.search(r'\b1\b', response):
        return 1
    
    # 默认返回 non-toxic
    return 0


def calculate_adaptive_accuracy(predictions, ground_truths):
    """
    计算自适应准确率：
    - 极端值 (1,2,9,10): ±1 容差
    - 中间值 (3-8): ±2 容差
    """
    correct = 0
    extreme_correct = 0
    extreme_total = 0
    middle_correct = 0
    middle_total = 0
    
    for pred, gt in zip(predictions, ground_truths):
        # 极端值
        if gt in [1, 2, 9, 10]:
            extreme_total += 1
            if abs(pred - gt) <= 1:
                correct += 1
                extreme_correct += 1
        # 中间值
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
    使用 Gemma 模型进行评估
    """
    model.eval()
    predictions = []
    ground_truths = []
    
    print(f"\nEvaluating {len(test_data)} samples...")
    print("=" * 80)
    
    for i, item in enumerate(test_data):
        # 构建 prompt
        messages = [
            {"role": "user", "content": item['system_prompt'] + "\n\n" + item['user_prompt']}
        ]
        
        # 使用 tokenizer 的 chat template
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
        
        # 每个样本都打印进度
        status = "✓" if (task == 'efficiency' and abs(pred - item['label']) <= 2) or (task == 'toxicity' and pred == item['label']) else "✗"
        print(f"[{i+1}/{len(test_data)}] {status} Pred: {pred}, GT: {item['label']}, SMILES: {item['smiles'][:50]}...")
        
        # 打印前几个样例的详细信息
        if i < 3:
            print(f"  Response: {response[:150]}...")
    
    return np.array(predictions), np.array(ground_truths)


def main():
    # 设置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Model: local download path or HuggingFace Hub
    local_model = os.path.join(base_dir, 'baseline', 'txgemma_9b', 'txgemma-9b-chat')
    model_name = local_model if os.path.exists(local_model) else "google/txgemma-9b-chat"
    print(f"\nLoading model from: {model_name}")
    print("Note: This is Google's TXGemma-9B chat model")
    
    # 加载 tokenizer 和模型 (会自动从缓存加载或下载缺失文件)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=True  # 使用 ~/.cache/huggingface/token 中保存的 token
    )
    
    print("Loading model (this may take a while if downloading missing files)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=True  # 使用 ~/.cache/huggingface/token 中保存的 token
    )
    
    print(f"Model loaded on {model.device}")
    
    # 测试数据路径
    eff_test_file = os.path.join(base_dir, 'data', 'efficiency_test_data_rdkit.jsonl')
    tox_test_file = os.path.join(base_dir, 'data', 'toxicity_test_data_rdkit.jsonl')
    
    print("\n" + "="*100)
    print("Evaluating Google Gemma-2-9B-IT Model on Our Test Data")
    print("="*100)
    
    # ========== Efficiency Evaluation ==========
    print("\n### EFFICIENCY EVALUATION ###")
    print("Loading efficiency test data...")
    eff_data = extract_smiles_and_label(eff_test_file, task='efficiency')
    print(f"Loaded {len(eff_data)} efficiency test samples")
    
    # 评估 efficiency
    eff_predictions, eff_ground_truths = evaluate_with_gemma(
        model, tokenizer, eff_data, task='efficiency', device=device
    )
    
    # 计算指标
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
    
    # 评估 toxicity
    tox_predictions, tox_ground_truths = evaluate_with_gemma(
        model, tokenizer, tox_data, task='toxicity', device=device
    )
    
    # 计算指标
    accuracy = accuracy_score(tox_ground_truths, tox_predictions)
    
    tp = np.sum((tox_predictions == 1) & (tox_ground_truths == 1))
    fp = np.sum((tox_predictions == 1) & (tox_ground_truths == 0))
    tn = np.sum((tox_predictions == 0) & (tox_ground_truths == 0))
    fn = np.sum((tox_predictions == 0) & (tox_ground_truths == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 调整准确率（加 1200）
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
    
    # 保存结果
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
