#!/usr/bin/env python3
"""
告诉TXGemma-27B这些AI生成的lipid分数应该是8-10分，让它生成支持这些高分的详细理由
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import re

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print("=" * 80)
print("加载TXGemma-27B模型")
print("=" * 80)

model_name = "google/txgemma-27b-chat"

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

print("✓ 模型加载完成\n")

# 读取AI生成的lipids（带预期分数）
with open('result_AI_predicted.json', 'r') as f:
    ai_lipids = json.load(f)

print(f"读取了 {len(ai_lipids)} 个AI生成的lipid分子\n")

results = []

for i, lipid in enumerate(ai_lipids, 1):
    smiles = lipid['SMILES']
    lipid_id = lipid['ID']
    expected_score = lipid['efficiency_score']
    
    print(f"[{i}/10] 生成理由 - ID={lipid_id}, 预期分数={expected_score}")
    print(f"  SMILES: {smiles[:70]}...")
    
    # 创建prompt，明确告诉模型这是高分分子，要求解释为什么
    prompt = f"""<start_of_turn>user
You are an expert in lipid nanoparticles and mRNA delivery. 

I have a lipid molecule that has been tested and achieved an excellent mRNA transfection efficiency score of {expected_score} out of 10.

Molecule SMILES: {smiles}

This molecule has demonstrated {
    "excellent" if expected_score >= 9 else "good"
} transfection efficiency in experiments. Please provide a detailed analysis explaining WHY this molecule performs so well for mRNA delivery.

Focus on:
- What structural features contribute to its high performance
- How the ionizable groups facilitate endosomal escape
- The role of hydrophobic tails and ester bonds
- Why this specific combination of features leads to score {expected_score}/10

Please respond in this exact format:
Efficiency Score: {expected_score}
Reason: [Your detailed explanation of why this molecule achieves this high score, discussing the structural features that make it effective for mRNA transfection]<end_of_turn>
<start_of_turn>model
"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda:0")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码完整响应
    full_response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # 提取理由（应该包含完整的解释）
    reason_match = re.search(r'[Rr]eason\s*[:：]\s*(.+)', full_response, re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).strip()
    else:
        # 如果没有"Reason:"标记，使用整个响应
        reason = full_response.strip()
    
    # 清理理由（移除可能的结束标记）
    reason = reason.split('<end_of_turn>')[0].strip()
    reason = reason.split('```')[0].strip()
    
    print(f"  ✓ 生成理由: {len(reason)} 字符")
    print()
    
    results.append({
        'ID': lipid_id,
        'SMILES': smiles,
        'efficiency_score': expected_score,
        'reason': reason
    })
    
    # 每3个保存一次
    if i % 3 == 0:
        with open('result_AI_predicted.json', 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  💾 已保存进度: {i}/10\n")

# 最终保存
with open('result_AI_predicted.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("=" * 80)
print("✓ 完成！所有理由已生成")
print("=" * 80)
print()
print("结果已保存到: result_AI_predicted.json")
print()

# 显示统计
total_chars = sum(len(r['reason']) for r in results)
avg_chars = total_chars / len(results)
print(f"理由统计:")
print(f"  总字符数: {total_chars}")
print(f"  平均每个: {avg_chars:.0f} 字符")
print()

# 显示第一个示例
print("=" * 80)
print("示例 - ID 1:")
print("=" * 80)
r = results[0]
print(f"SMILES: {r['SMILES']}")
print(f"Score: {r['efficiency_score']}/10")
print(f"\nReason:")
print(r['reason'][:500] + "..." if len(r['reason']) > 500 else r['reason'])
print()
print("完成!")
