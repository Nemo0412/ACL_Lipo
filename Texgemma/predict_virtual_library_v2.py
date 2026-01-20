#!/usr/bin/env python3
"""
使用 Google Gemma-2-27B-IT 模型预测虚拟分子库的 mRNA 转染效率
优化版：使用预处理数据 + 详细的结构特征
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
    
    if len(data) > 0:
        print(f"  Sample fields: {list(data[0].keys())}")
    
    return data


def create_enhanced_prompt(mol_data):
    """
    创建增强版 prompt，包含详细的结构特征信息
    """
    system_prompt = """You are an expert in lipid nanoparticle (LNP) design for mRNA delivery with deep knowledge of:
- Structure-activity relationships in ionizable lipids
- Factors affecting transfection efficiency
- Molecular features that optimize mRNA encapsulation and delivery

Your task is to predict mRNA transfection efficiency and provide detailed mechanistic rationale."""

    user_prompt = f"""Analyze this lipid molecule and predict its mRNA transfection efficiency.

**Molecular Information:**
- SMILES Structure: {mol_data['SMILES']}
- Amino Acid Base: {mol_data['amino_acid']}
- Protection Group: {mol_data['protection_group']}
- Linker Length: {mol_data['linker_length']} carbons
- OCOO Ester Bonds: {mol_data['ester_bonds_OCOO']}
- Number of Tails: {mol_data['tail_count']}
- Estimated Total Carbons: ~{mol_data.get('estimated_total_carbons', 'N/A')}

**Summary:** {mol_data['features_description']}

**Please provide:**

1. **Efficiency Score** (1-10 scale):
   - 1-2: Very poor transfection
   - 3-4: Poor transfection
   - 5-6: Moderate transfection
   - 7-8: Good transfection
   - 9-10: Excellent transfection

2. **Detailed Rationale** addressing:
   
   a) **Hydrophobic Tail Length**: How does the tail structure affect:
      - Membrane interaction and fusion
      - LNP stability
      - Endosomal escape
   
   b) **Ester Bond Type and Position**: Impact on:
      - Biodegradability
      - pH sensitivity
      - Clearance rate
   
   c) **Number of Lipid Tails**: Effect on:
      - Molecular geometry and packing
      - Membrane curvature
      - Self-assembly properties
   
   d) **Tertiary Amines/Ionizable Groups**: Influence on:
      - Proton buffering capacity
      - Endosomal escape efficiency
      - Cytotoxicity profile
   
   e) **Overall Structure Balance**: Integration of all factors

**Output Format:**
Efficiency Score: [1-10]

Rationale:
[Your detailed mechanistic explanation addressing all factors above]

**Important**: Be specific about WHY each structural feature contributes to the predicted efficiency. Avoid generic statements."""

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
        r'[Pp]redicted\s+[Ss]core\s*[:：]\s*(\d+)',
        r'(\d+)\s*/\s*10',
        r'(\d+)\s*out\s+of\s+10',
        r'[Rr]ating\s*[:：]\s*(\d+)',
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
        r'[Aa]nalysis\s*[:：]\s*(.+?)(?=\n\n|\Z)',
    ]
    
    for pattern in rationale_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            rationale = match.group(1).strip()
            break
    
    # 如果没有明确的理由部分，使用整个响应（去掉分数部分）
    if not rationale:
        lines = response.split('\n')
        rationale_lines = [line for line in lines if not re.search(r'[Ss]core\s*[:：]', line)]
        rationale = '\n'.join(rationale_lines).strip()
    
    # 如果还是没有分数，默认为5
    if score is None:
        score = 5
        rationale = f"[WARNING: Model response did not contain a clear score, defaulting to 5]\n\n{rationale}"
    
    return score, rationale


def predict_with_gemma(model, tokenizer, molecules, device='cuda', batch_size=1):
    """
    使用 Gemma-2-27B-IT 模型预测 mRNA 转染效率
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
            system_prompt, user_prompt = create_enhanced_prompt(mol_data)
            
            messages = [
                {"role": "user", "content": system_prompt + "\n\n" + user_prompt}
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
                    max_new_tokens=768,  # 更长的输出以包含详细理由
                    do_sample=True,
                    temperature=0.7,    # 增加多样性，避免总预测同一个值
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # 提取分数和理由
            score, rationale = extract_score_and_rationale(response)
            
            result = {
                'ID': mol_id,
                'SMILES': mol_data['SMILES'],
                'amino_acid': mol_data['amino_acid'],
                'protection_group': mol_data['protection_group'],
                'linker_length': mol_data['linker_length'],
                'ester_bonds': mol_data['ester_bonds_OCOO'],
                'tail_count': mol_data['tail_count'],
                'efficiency_score': score,
                'rationale': rationale,
                'full_response': response
            }
            
            results.append(result)
            
            # 每处理50个分子打印一次进度详情
            if (i + 1) % 50 == 0:
                print(f"\n[{i+1}/{len(molecules)}] Progress Update:")
                print(f"  Last molecule: ID={mol_id}, Score={score}")
                print(f"  Rationale preview: {rationale[:150]}...")
                
                # 当前分数分布
                if len(results) >= 50:
                    score_dist = {}
                    for r in results[-50:]:
                        s = r['efficiency_score']
                        score_dist[s] = score_dist.get(s, 0) + 1
                    print(f"  Last 50 score distribution: {dict(sorted(score_dist.items()))}")
        
        except Exception as e:
            print(f"\n✗ Error processing molecule {mol_id}: {e}")
            # 添加失败的结果
            result = {
                'ID': mol_id,
                'SMILES': mol_data['SMILES'],
                'amino_acid': mol_data['amino_acid'],
                'protection_group': mol_data['protection_group'],
                'linker_length': mol_data['linker_length'],
                'ester_bonds': mol_data['ester_bonds_OCOO'],
                'tail_count': mol_data['tail_count'],
                'efficiency_score': 5,  # 默认值
                'rationale': f"ERROR: {str(e)}",
                'full_response': ""
            }
            results.append(result)
    
    return results


def main():
    # 设置路径
    data_dir = '/mnt/3fs/dots-pretrain/leshu/workspace/Texgemma/data'
    input_file = os.path.join(data_dir, 'virtual_library_preprocessed.jsonl')
    output_dir = '/mnt/3fs/dots-pretrain/leshu/workspace/Texgemma'
    
    # 检查预处理数据是否存在
    if not os.path.exists(input_file):
        print(f"✗ Preprocessed data not found: {input_file}")
        print(f"\nPlease run data preprocessing first:")
        print(f"  python3 preprocess_data.py")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载预处理数据
    print("\n" + "="*100)
    print("Loading Preprocessed Virtual Library Data")
    print("="*100)
    
    molecules = load_preprocessed_data(input_file)
    
    # 加载模型
    print("\n" + "="*100)
    print("Loading Gemma-2-27B-IT Model")
    print("="*100)
    
    model_name = "google/gemma-2-27b-it"
    
    print(f"\nModel: {model_name}")
    print("Note: Using Gemma-2-27B-IT (instruction-tuned chat model)")
    print("This model is optimized for following complex instructions and providing detailed explanations.")
    print("\nLoading may take several minutes...")
    
    try:
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
        
        print(f"\n✓ Model loaded successfully")
        print(f"  Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'auto'}")
        
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check HuggingFace token: cat ~/.cache/huggingface/token")
        print("2. Check model exists: huggingface-cli repo info google/gemma-2-27b-it")
        print("3. Try downloading manually: huggingface-cli download google/gemma-2-27b-it")
        return
    
    # 开始预测
    print("\n" + "="*100)
    print("Starting Predictions")
    print("="*100)
    print(f"Total molecules: {len(molecules)}")
    print(f"Estimated time: ~{len(molecules) * 12 / 3600:.1f} hours (assuming 12s per molecule)")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = datetime.now()
    
    results = predict_with_gemma(
        model=model,
        tokenizer=tokenizer,
        molecules=molecules,
        device=device
    )
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print(f"\n✓ Predictions completed!")
    print(f"  Total time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
    print(f"  Average time per molecule: {elapsed/len(molecules):.2f} seconds")
    print(f"  End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 按照效率分数从大到小排序
    print("\n" + "="*100)
    print("Sorting Results by Efficiency Score (High to Low)")
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
    
    # 保存完整结果（包含理由）
    output_file_full = os.path.join(output_dir, 'virtual_library_predictions_full.json')
    with open(output_file_full, 'w', encoding='utf-8') as f:
        json.dump(results_sorted, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Full results saved to: {output_file_full}")
    
    # 保存简化版本（只有关键信息）
    results_simplified = [
        {
            'ID': r['ID'],
            'SMILES': r['SMILES'],
            'efficiency_score': r['efficiency_score'],
            'rationale': r['rationale'],
            'structure_features': {
                'amino_acid': r['amino_acid'],
                'protection_group': r['protection_group'],
                'linker_length': r['linker_length'],
                'ester_bonds': r['ester_bonds'],
                'tail_count': r['tail_count']
            }
        }
        for r in results_sorted
    ]
    
    output_file_simple = os.path.join(output_dir, 'result.json')
    with open(output_file_simple, 'w', encoding='utf-8') as f:
        json.dump(results_simplified, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Simplified results saved to: {output_file_simple}")
    
    # 保存 Top 100 分子
    top100 = results_sorted[:100]
    output_file_top100 = os.path.join(output_dir, 'top100_high_efficiency.json')
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
        print(f"  Structure: {result['amino_acid']}-based, {result['protection_group']}, "
              f"{result['tail_count']} tails, linker={result['linker_length']}")
        print(f"  SMILES: {result['SMILES'][:80]}...")
        print(f"  Rationale: {result['rationale'][:250]}...")
    
    # 保存运行日志
    log_file = os.path.join(output_dir, 'prediction_summary.txt')
    with open(log_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("TXGemma Virtual Library Prediction Summary\n")
        f.write("="*100 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Total molecules: {len(results_sorted)}\n")
        f.write(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {elapsed/3600:.2f} hours\n")
        f.write(f"Average time: {elapsed/len(molecules):.2f} seconds/molecule\n\n")
        f.write("Score Distribution:\n")
        for score in sorted(score_dist.keys(), reverse=True):
            count = score_dist[score]
            percentage = count / len(results_sorted) * 100
            f.write(f"  Score {score:2d}: {count:4d} molecules ({percentage:5.1f}%)\n")
    
    print(f"\n✓ Summary log saved to: {log_file}")
    
    print("\n" + "="*100)
    print("Prediction Complete!")
    print("="*100)
    print(f"📁 Output files:")
    print(f"   1. result.json - Simplified results (sorted by score)")
    print(f"   2. virtual_library_predictions_full.json - Complete results with full responses")
    print(f"   3. top100_high_efficiency.json - Top 100 high-efficiency molecules")
    print(f"   4. prediction_summary.txt - Run statistics")
    print(f"\n📊 Recommended next steps:")
    print(f"   - Review Top 100 molecules for experimental validation")
    print(f"   - Analyze score distribution and structural patterns")
    print(f"   - Compare rationales to identify key structural features")


if __name__ == '__main__':
    main()
