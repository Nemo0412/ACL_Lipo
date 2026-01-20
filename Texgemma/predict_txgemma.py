#!/usr/bin/env python3
"""
使用 Google TXGemma-27B-Predict 模型预测虚拟分子库的 mRNA 转染效率
按照用户指定的 prompt 格式
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from tqdm import tqdm
import os
from datetime import datetime

def load_preprocessed_data(jsonl_file):
    """
    加载预处理后的 JSONL 数据
    """
    print(f"Loading preprocessed data from: {jsonl_file}")
    
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"✓ Loaded {len(data)} molecules")
    return data


def create_prediction_prompt(mol_data):
    """
    创建预测 prompt - 使用对话格式
    """
    # 对话模型格式的 prompt
    conversation = f"""<start_of_turn>user
You are an expert in lipid nanoparticles and mRNA delivery. Please predict the mRNA transfection efficiency for the following lipid molecule.

Molecule SMILES: {mol_data['SMILES']}

Structure Features:
- Amino acid base: {mol_data['amino_acid']}
- Protection group: {mol_data['protection_group']}
- Linker length: {mol_data['linker_length']} carbons
- Number of lipid tails: {mol_data['tail_count']}
- OCOO ester bonds: {mol_data['ester_bonds_OCOO']}

Please provide not only the predicted performance but also the underlying rationale, with specific attention to how factors such as hydrophobic tail length, ester bond type, number of lipid tails, and number of tertiary amines may influence mRNA transfection efficiency.

Provide an efficiency score from 1 to 10, where:
- 1-2: Very poor transfection
- 3-4: Poor transfection  
- 5-6: Moderate transfection
- 7-8: Good transfection
- 9-10: Excellent transfection

Please respond in this exact format:
Efficiency Score: [number from 1-10]
Reason: [Your detailed analysis of the molecular structure and its impact on transfection efficiency]<end_of_turn>
<start_of_turn>model
"""

    return conversation


def extract_score_and_reason(response):
    """
    从模型响应中提取效率分数和理由
    """
    score = None
    reason = ""
    
    # 提取分数
    score_patterns = [
        r'[Ee]fficiency\s+[Ss]core\s*[:：]\s*(\d+)',
        r'[Ss]core\s*[:：]\s*(\d+)',
        r'[Pp]redicted\s+[Ss]core\s*[:：]\s*(\d+)',
        r'(\d+)\s*/\s*10',
        r'(\d+)\s*out\s+of\s+10',
    ]
    
    for pattern in score_patterns:
        match = re.search(pattern, response)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 10:
                break
    
    # 如果没找到明确分数，尝试找任何1-10的数字
    if score is None:
        numbers = re.findall(r'\b([1-9]|10)\b', response)
        if numbers:
            score = int(numbers[0])
    
    # 提取理由
    reason_patterns = [
        r'[Rr]eason\s*[:：]\s*(.+?)(?=\n\n|\Z)',
        r'[Rr]ationale\s*[:：]\s*(.+?)(?=\n\n|\Z)',
        r'[Ee]xplanation\s*[:：]\s*(.+?)(?=\n\n|\Z)',
    ]
    
    for pattern in reason_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            reason = match.group(1).strip()
            break
    
    # 如果没有明确的理由部分，使用整个响应（去掉分数行）
    if not reason:
        lines = response.split('\n')
        reason_lines = [line for line in lines if not re.search(r'[Ss]core\s*[:：]', line)]
        reason = '\n'.join(reason_lines).strip()
    
    # 默认分数
    if score is None:
        score = 5
        reason = f"[WARNING: No clear score found in response, defaulting to 5]\n\n{reason}"
    
    return score, reason


def predict_batch(model, tokenizer, molecules, device='cuda'):
    """
    批量预测分子的 mRNA 转染效率
    """
    model.eval()
    results = []
    
    print(f"\n{'='*100}")
    print(f"Starting Prediction for {len(molecules)} molecules")
    print(f"{'='*100}\n")
    
    for i, mol_data in enumerate(tqdm(molecules, desc="Predicting")):
        mol_id = mol_data['ID']
        
        try:
            # 构建 prompt
            prompt = create_prediction_prompt(mol_data)
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # 提取分数和理由
            score, reason = extract_score_and_reason(response)
            
            result = {
                'ID': mol_id,
                'SMILES': mol_data['SMILES'],
                'efficiency_score': score,
                'reason': reason
            }
            
            results.append(result)
            
            # 每50个分子打印进度
            if (i + 1) % 50 == 0:
                print(f"\n[{i+1}/{len(molecules)}] Processed {i+1} molecules")
                print(f"  Last: ID={mol_id}, Score={score}")
                
                # 显示最近50个的分数分布
                recent_scores = [r['efficiency_score'] for r in results[-50:]]
                score_dist = {s: recent_scores.count(s) for s in set(recent_scores)}
                print(f"  Recent 50 score distribution: {dict(sorted(score_dist.items()))}")
        
        except Exception as e:
            print(f"\n✗ Error processing molecule {mol_id}: {e}")
            result = {
                'ID': mol_id,
                'SMILES': mol_data['SMILES'],
                'efficiency_score': 5,
                'reason': f"ERROR: {str(e)}"
            }
            results.append(result)
    
    return results


def main():
    # 路径设置
    data_file = '/mnt/3fs/dots-pretrain/leshu/workspace/Texgemma/data/virtual_library_preprocessed.jsonl'
    output_file = '/mnt/3fs/dots-pretrain/leshu/workspace/Texgemma/result.json'
    
    # 检查数据文件
    if not os.path.exists(data_file):
        print(f"✗ Data file not found: {data_file}")
        print("Please run preprocess_data.py first")
        return
    
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    print("\n" + "="*100)
    print("Loading Data")
    print("="*100)
    molecules = load_preprocessed_data(data_file)
    
    # 加载模型
    print("\n" + "="*100)
    print("Loading TXGemma-27B-Predict Model")
    print("="*100)
    
    # 使用 txgemma-27b-chat 对话模型
    model_names = [
        "google/txgemma-27b-chat",
        "google/gemma-2-27b-it"
    ]
    
    model = None
    tokenizer = None
    model_name_used = None
    
    for model_name in model_names:
        try:
            print(f"\nTrying to load: {model_name}")
            
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
            
            model_name_used = model_name
            print(f"✓ Successfully loaded: {model_name}")
            break
            
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")
            if model_name == model_names[-1]:
                print("\n✗ All model options failed!")
                return
    
    # 开始预测
    print("\n" + "="*100)
    print("Starting Predictions")
    print("="*100)
    print(f"Model: {model_name_used}")
    print(f"Total molecules: {len(molecules)}")
    print(f"Estimated time: ~{len(molecules) * 10 / 3600:.1f} hours")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = datetime.now()
    
    results = predict_batch(
        model=model,
        tokenizer=tokenizer,
        molecules=molecules,
        device=device
    )
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print(f"\n✓ Predictions completed!")
    print(f"  Total time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
    print(f"  Average: {elapsed/len(molecules):.2f} seconds/molecule")
    
    # 按照 efficiency_score 从大到小排序
    print("\n" + "="*100)
    print("Sorting by Efficiency Score (High to Low)")
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
        bar = '█' * int(percentage / 2)
        print(f"  Score {score:2d}: {count:4d} molecules ({percentage:5.1f}%) {bar}")
    
    # 保存结果到 result.json
    print(f"\nSaving results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_sorted, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved successfully")
    
    # 显示 Top 10
    print("\n" + "="*100)
    print("Top 10 Molecules by Efficiency Score")
    print("="*100)
    
    for i, result in enumerate(results_sorted[:10], 1):
        print(f"\n#{i} - Molecule ID: {result['ID']}")
        print(f"  Efficiency Score: {result['efficiency_score']}/10")
        print(f"  SMILES: {result['SMILES'][:80]}...")
        print(f"  Reason: {result['reason'][:200]}...")
    
    # 保存统计摘要
    summary_file = output_file.replace('.json', '_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("TXGemma Virtual Library Prediction Summary\n")
        f.write("="*100 + "\n\n")
        f.write(f"Model: {model_name_used}\n")
        f.write(f"Total molecules: {len(results_sorted)}\n")
        f.write(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {elapsed/3600:.2f} hours\n")
        f.write(f"Average: {elapsed/len(molecules):.2f} seconds/molecule\n\n")
        f.write("Score Distribution:\n")
        for score in sorted(score_dist.keys(), reverse=True):
            count = score_dist[score]
            percentage = count / len(results_sorted) * 100
            f.write(f"  Score {score:2d}: {count:4d} molecules ({percentage:5.1f}%)\n")
        f.write("\n\nTop 10 Molecules:\n")
        for i, result in enumerate(results_sorted[:10], 1):
            f.write(f"\n#{i} - ID: {result['ID']}, Score: {result['efficiency_score']}/10\n")
            f.write(f"  SMILES: {result['SMILES']}\n")
            f.write(f"  Reason: {result['reason'][:500]}...\n")
    
    print(f"\n✓ Summary saved to: {summary_file}")
    
    print("\n" + "="*100)
    print("Prediction Complete!")
    print("="*100)
    print(f"📁 Output files:")
    print(f"   - result.json (sorted by efficiency score, high to low)")
    print(f"   - result_summary.txt (statistics and top 10)")
    print(f"\n📊 Total molecules: {len(results_sorted)}")
    print(f"📈 Score range: {min(r['efficiency_score'] for r in results_sorted)} - {max(r['efficiency_score'] for r in results_sorted)}")


if __name__ == '__main__':
    main()
