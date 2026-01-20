#!/usr/bin/env python3
"""
使用 Google TXGemma-27B-Predict 模型预测虚拟分子库的 mRNA 转染效率
"""

import pandas as pd
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from tqdm import tqdm
import os
from datetime import datetime

def load_virtual_library(excel_file):
    """
    加载虚拟分子库数据
    """
    print(f"Loading virtual library from: {excel_file}")
    df = pd.read_excel(excel_file)
    
    print(f"Loaded {len(df)} molecules")
    print(f"Columns: {df.columns.tolist()}")
    
    # 检查 SMILES 列名
    smiles_col = None
    for col in df.columns:
        if 'smiles' in col.lower() or 'SMILES' in col:
            smiles_col = col
            break
    
    if smiles_col is None:
        raise ValueError(f"Cannot find SMILES column in {df.columns.tolist()}")
    
    print(f"Using SMILES column: {smiles_col}")
    
    # 添加 ID 列（如果没有）
    if 'ID' not in df.columns and 'id' not in df.columns:
        df['ID'] = range(1, len(df) + 1)
    
    return df, smiles_col


def create_prediction_prompt(smiles):
    """
    创建预测 prompt，要求模型提供预测分数和理由
    """
    system_prompt = """You are an expert in medicinal chemistry and lipid nanoparticle (LNP) design for mRNA delivery. 
You have deep knowledge of structure-activity relationships and how molecular features affect transfection efficiency."""

    user_prompt = f"""Please predict the mRNA transfection efficiency for the following lipid molecule and provide a detailed rationale.

Molecule SMILES: {smiles}

Please provide:
1. An efficiency score from 1 to 10 (where 1 is very poor and 10 is excellent)
2. A detailed explanation of your prediction, specifically addressing how the following factors influence the transfection efficiency:
   - Hydrophobic tail length
   - Ester bond type and position
   - Number of lipid tails
   - Number of tertiary amines
   - Overall molecular structure and balance

Please provide not only the predicted performance but also the underlying rationale, with specific attention to how factors such as hydrophobic tail length, ester bond type, number of lipid tails, and number of tertiary amines may influence mRNA transfection efficiency.

Format your response as:
Efficiency Score: [1-10]
Rationale: [Your detailed explanation]"""

    return system_prompt, user_prompt


def extract_score_and_rationale(response):
    """
    从模型响应中提取效率分数和理由
    """
    score = None
    rationale = ""
    
    # 提取分数
    score_patterns = [
        r'[Ee]fficiency\s+[Ss]core\s*[:：]\s*(\d+)',
        r'[Ss]core\s*[:：]\s*(\d+)',
        r'[Pp]redicted\s+score\s*[:：]\s*(\d+)',
        r'(\d+)\s*/\s*10',
        r'(\d+)\s*out\s+of\s+10',
    ]
    
    for pattern in score_patterns:
        match = re.search(pattern, response)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 10:
                break
    
    # 如果没找到，尝试找任何1-10的数字
    if score is None:
        numbers = re.findall(r'\b([1-9]|10)\b', response)
        if numbers:
            score = int(numbers[0])
    
    # 提取理由
    rationale_patterns = [
        r'[Rr]ationale\s*[:：]\s*(.+?)(?=\n\n|\Z)',
        r'[Ee]xplanation\s*[:：]\s*(.+?)(?=\n\n|\Z)',
        r'[Rr]eason\s*[:：]\s*(.+?)(?=\n\n|\Z)',
    ]
    
    for pattern in rationale_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            rationale = match.group(1).strip()
            break
    
    # 如果没有明确的理由部分，使用整个响应（去掉分数部分）
    if not rationale:
        # 移除分数相关的行
        lines = response.split('\n')
        rationale_lines = [line for line in lines if not re.search(r'[Ss]core\s*[:：]', line)]
        rationale = '\n'.join(rationale_lines).strip()
    
    # 如果还是没有分数，默认为5
    if score is None:
        score = 5
        rationale = f"[Model response did not contain a clear score] {rationale}"
    
    return score, rationale


def predict_with_txgemma(model, tokenizer, smiles_list, device='cuda', batch_size=1):
    """
    使用 TXGemma-27B 模型预测 mRNA 转染效率
    """
    model.eval()
    results = []
    
    print(f"\nPredicting {len(smiles_list)} molecules...")
    print("=" * 100)
    
    for i, row in enumerate(tqdm(smiles_list, desc="Predicting")):
        mol_id = row.get('ID', i + 1)
        smiles = row['SMILES']
        
        # 构建 prompt
        system_prompt, user_prompt = create_prediction_prompt(smiles)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 使用 tokenizer 的 chat template
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            # 如果没有 chat template，使用简单拼接
            prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  # 更长的输出以包含详细理由
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # 提取分数和理由
        score, rationale = extract_score_and_rationale(response)
        
        result = {
            'ID': mol_id,
            'SMILES': smiles,
            'efficiency_score': score,
            'rationale': rationale,
            'full_response': response
        }
        
        results.append(result)
        
        # 每处理100个分子打印一次进度
        if (i + 1) % 100 == 0:
            print(f"\n[{i+1}/{len(smiles_list)}] Processed {i+1} molecules")
            print(f"  Last prediction: ID={mol_id}, Score={score}")
            print(f"  Rationale preview: {rationale[:100]}...")
    
    return results


def main():
    # 设置路径
    data_dir = '/mnt/3fs/dots-pretrain/leshu/workspace/Txgemma/data'
    input_file = os.path.join(data_dir, '10000-virtual library.xlsx')
    output_dir = '/mnt/3fs/dots-pretrain/leshu/workspace/Texgemma'
    
    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    print("\n" + "="*100)
    print("Loading Virtual Library Data")
    print("="*100)
    
    df, smiles_col = load_virtual_library(input_file)
    
    # 准备 SMILES 列表
    smiles_data = []
    for idx, row in df.iterrows():
        smiles_data.append({
            'ID': row.get('ID', idx + 1),
            'SMILES': row[smiles_col]
        })
    
    # 加载模型
    print("\n" + "="*100)
    print("Loading TXGemma-27B-Predict Model")
    print("="*100)
    
    model_name = "google/txgemma-27b-predict"
    
    print(f"Model: {model_name}")
    print("Note: This is a large model, loading may take several minutes...")
    
    try:
        # 尝试从本地或在线加载
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=True  # 使用保存的 HF token
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=True
        )
        
        print(f"✓ Model loaded successfully on {device}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying alternative model name: google/gemma-2-27b-it")
        
        # 尝试使用 Gemma-2-27B-IT (如果 txgemma-27b-predict 不存在)
        model_name = "google/gemma-2-27b-it"
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=True
        )
        
        print(f"✓ Alternative model loaded: {model_name}")
    
    # 开始预测
    print("\n" + "="*100)
    print("Starting Predictions")
    print("="*100)
    print(f"Total molecules to predict: {len(smiles_data)}")
    print(f"Estimated time: ~{len(smiles_data) * 10 / 3600:.1f} hours (assuming 10s per molecule)")
    
    start_time = datetime.now()
    
    results = predict_with_txgemma(
        model=model,
        tokenizer=tokenizer,
        smiles_list=smiles_data,
        device=device
    )
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print(f"\n✓ Predictions completed!")
    print(f"  Total time: {elapsed/3600:.2f} hours")
    print(f"  Average time per molecule: {elapsed/len(smiles_data):.2f} seconds")
    
    # 按照效率分数从大到小排序
    print("\n" + "="*100)
    print("Sorting Results by Efficiency Score")
    print("="*100)
    
    results_sorted = sorted(results, key=lambda x: x['efficiency_score'], reverse=True)
    
    # 统计分数分布
    score_dist = {}
    for r in results_sorted:
        score = r['efficiency_score']
        score_dist[score] = score_dist.get(score, 0) + 1
    
    print("\nScore Distribution:")
    for score in sorted(score_dist.keys(), reverse=True):
        count = score_dist[score]
        percentage = count / len(results_sorted) * 100
        print(f"  Score {score}: {count} molecules ({percentage:.1f}%)")
    
    # 保存完整结果（包含理由）
    output_file = os.path.join(output_dir, 'virtual_library_predictions_with_rationale.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_sorted, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Full results saved to: {output_file}")
    
    # 保存简化版本（只有分数和理由，不包含完整响应）
    results_simplified = [
        {
            'ID': r['ID'],
            'SMILES': r['SMILES'],
            'efficiency_score': r['efficiency_score'],
            'rationale': r['rationale']
        }
        for r in results_sorted
    ]
    
    output_file_simple = os.path.join(output_dir, 'result.json')
    with open(output_file_simple, 'w', encoding='utf-8') as f:
        json.dump(results_simplified, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Simplified results saved to: {output_file_simple}")
    
    # 保存 Top 100 分子（详细版）
    top100 = results_sorted[:100]
    output_file_top100 = os.path.join(output_dir, 'top100_predictions.json')
    with open(output_file_top100, 'w', encoding='utf-8') as f:
        json.dump(top100, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Top 100 molecules saved to: {output_file_top100}")
    
    # 打印 Top 10
    print("\n" + "="*100)
    print("Top 10 Molecules by Efficiency Score")
    print("="*100)
    
    for i, result in enumerate(top100[:10], 1):
        print(f"\n#{i} - ID: {result['ID']}")
        print(f"  Score: {result['efficiency_score']}/10")
        print(f"  SMILES: {result['SMILES'][:80]}...")
        print(f"  Rationale: {result['rationale'][:200]}...")
    
    print("\n" + "="*100)
    print("Prediction Complete!")
    print("="*100)
    print(f"Total molecules predicted: {len(results_sorted)}")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
