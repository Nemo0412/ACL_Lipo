#!/usr/bin/env python3
"""
为AI生成的10个lipid重新预测，保存完整响应
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import re

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print("加载模型...")
model_name = "google/txgemma-27b-chat"

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=True)

print("✓ 模型加载完成\n")

# 读取AI生成的lipids
with open('result_AI.json', 'r') as f:
    ai_lipids = json.load(f)

results = []

for i, lipid in enumerate(ai_lipids, 1):
    smiles = lipid['smiles']
    lipid_id = lipid['id']
    
    print(f"[{i}/10] 预测 ID={lipid_id}")
    print(f"  SMILES: {smiles[:60]}...")
    
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
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda:0")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,  # 更长的输出
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码完整响应
    full_response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # 提取分数
    score = 5  # 默认
    score_match = re.search(r'[Ee]fficiency\s+[Ss]core\s*[:：]\s*(\d+)', full_response)
    if score_match:
        score = int(score_match.group(1))
    
    # 提取理由（保留完整内容）
    reason_match = re.search(r'[Rr]eason\s*[:：]\s*(.+)', full_response, re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).strip()
    else:
        reason = full_response.strip()
    
    print(f"  Score: {score}, Reason: {len(reason)} chars")
    
    results.append({
        'ID': lipid_id,
        'SMILES': smiles,
        'efficiency_score': score,
        'reason': reason
    })
    
    # 每3个保存一次
    if i % 3 == 0:
        with open('result_AI_predicted.json', 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  [保存进度: {i}/10]\n")

# 最终保存
with open('result_AI_predicted.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n✓ 完成！结果已保存到 result_AI_predicted.json")

# 统计
score_dist = {}
for r in results:
    s = r['efficiency_score']
    score_dist[s] = score_dist.get(s, 0) + 1

print("\n分数分布:")
for s in sorted(score_dist.keys()):
    print(f"  分数 {s}: {score_dist[s]} 个")
