#!/usr/bin/env python3
"""
使用TXGemma模型为AI生成的10个lipid SMILES预测效率分数和详细理由
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import re

# 设置离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print("=" * 80)
print("加载TXGemma-27B模型")
print("=" * 80)

model_name = "google/txgemma-27b-chat"
device = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    local_files_only=True,
    trust_remote_code=True,
    use_fast=False
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    trust_remote_code=True
)

print(f"✓ 模型加载完成")
print()

# 读取AI生成的lipids
with open('result_AI.json', 'r') as f:
    ai_lipids = json.load(f)

print(f"读取了 {len(ai_lipids)} 个AI生成的lipid分子")
print()

def create_prediction_prompt(smiles):
    """创建预测prompt"""
    prompt = f"""<start_of_turn>user
You are an expert in lipid nanoparticles and mRNA delivery. Please predict the mRNA transfection efficiency for the following lipid molecule.

Molecule SMILES: {smiles}

Please provide not only the predicted performance but also the underlying rationale, with specific attention to how structural features such as ionizable groups, hydrophobic tail length, ester bonds, and other factors may influence mRNA transfection efficiency.

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
    return prompt

def extract_score_and_reason(response):
    """从响应中提取分数和理由"""
    score = None
    reason = ""
    
    # 提取分数
    score_patterns = [
        r'[Ee]fficiency\s+[Ss]core\s*[:：]\s*(\d+)',
        r'[Ss]core\s*[:：]\s*(\d+)',
    ]
    
    for pattern in score_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 10:
                break
    
    # 如果没找到分数，设置默认值
    if score is None:
        score = 5
    
    # 提取理由 - 使用re.DOTALL来匹配多行
    reason_match = re.search(r'[Rr]eason\s*[:：]\s*(.+)', response, re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).strip()
    else:
        # 如果找不到"Reason:"标记，就使用整个响应
        reason = response.strip()
    
    return score, reason

# 存储结果
results = []

print("=" * 80)
print("开始预测效率分数和理由")
print("=" * 80)
print()

for i, lipid in enumerate(ai_lipids, 1):
    smiles = lipid['smiles']
    original_id = lipid['id']
    
    print(f"[{i}/10] 预测 Lipid ID={original_id}...")
    print(f"  SMILES: {smiles[:60]}...")
    
    # 创建prompt
    prompt = create_prediction_prompt(smiles)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    
    # Generate (使用更长的max_new_tokens来获取完整理由)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=800,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # 提取分数和理由
    score, reason = extract_score_and_reason(response)
    
    print(f"  ✓ Score: {score}, Reason length: {len(reason)} chars")
    print()
    
    # 保存结果
    result = {
        'ID': original_id,
        'SMILES': smiles,
        'efficiency_score': score,
        'reason': reason
    }
    results.append(result)
    
    # 每3个保存一次
    if i % 3 == 0:
        with open('result_AI_predicted.json', 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  [已保存进度: {i}/10]")
        print()

# 最终保存
with open('result_AI_predicted.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("=" * 80)
print("预测完成！")
print("=" * 80)
print()
print("结果已保存到: result_AI_predicted.json")
print()

# 显示分数分布
score_dist = {}
for r in results:
    score = r['efficiency_score']
    score_dist[score] = score_dist.get(score, 0) + 1

print("分数分布:")
for score in sorted(score_dist.keys()):
    count = score_dist[score]
    print(f"  分数 {score}: {count} 个")
print()

# 显示前3个示例
print("前3个预测示例:")
for r in results[:3]:
    print(f"\nID {r['ID']}: Score={r['efficiency_score']}")
    print(f"SMILES: {r['SMILES'][:60]}...")
    print(f"Reason: {r['reason'][:150]}...")

print("\n完成!")
