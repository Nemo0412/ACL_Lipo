#!/usr/bin/env python3
"""
使用TXGemma模型生成10个高效的、可合成的ionizable lipid SMILES（目标9-10分）
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os

# 设置离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print("="*80)
print("使用TXGemma生成高效ionizable lipid SMILES")
print("="*80)

# 加载模型
model_name = "google/txgemma-27b-chat"
print(f"\nLoading model: {model_name}")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    local_files_only=True,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    local_files_only=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

print("✓ Model loaded successfully")

# 创建生成提示
generation_prompt = """<start_of_turn>user
You are an expert in ionizable lipid design for mRNA delivery. Design 10 novel ionizable lipid molecules with excellent mRNA transfection efficiency (predicted score 9-10 out of 10).

IMPORTANT REQUIREMENTS:
1. Provide SINGLE, valid SMILES strings (no salts, no ions, no dots separating components)
2. Each lipid should be a single neutral molecule containing:
   - Tertiary amine group (ionizable, pKa 6-7)
   - Biodegradable ester linkages
   - 2-4 hydrophobic alkyl tails (C12-C18)
   - Branched or asymmetric structure

3. Base your designs on successful examples like:
   - DLin-MC3-DMA: Branched tails with tertiary amine
   - ALC-0315: Asymmetric structure with different tail lengths
   - SM-102: Contains cyclic amine with multiple tails

For EACH of the 10 designs, provide EXACTLY this format:

Design [1-10]:
SMILES: [single valid SMILES string without dots or charges]
Description: [One sentence describing the key features]
Expected Score: 9-10

Make the SMILES strings realistic and synthesizable. Focus on structural diversity while maintaining high predicted transfection efficiency.

<end_of_turn>
<start_of_turn>model
"""

print("\n" + "="*80)
print("Generating 10 high-efficiency ionizable lipid designs...")
print("="*80 + "\n")

# Generate
inputs = tokenizer(generation_prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=2500,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print("Raw Model Response:")
print("="*80)
print(response)
print("="*80)

# 保存原始响应
with open('generation_raw_response.txt', 'w') as f:
    f.write(response)

print("\n✓ Raw response saved to generation_raw_response.txt")

# 解析响应
import re

results = []
designs = response.split('Design ')

for design_text in designs[1:]:  # Skip first empty split
    lines = design_text.strip().split('\n')
    
    design = {}
    design_num = None
    
    for line in lines:
        line = line.strip()
        
        # Extract design number
        if not design_num and ':' in line:
            design_num = line.split(':')[0].strip()
            design['ID'] = design_num
        
        # Extract SMILES
        if line.startswith('SMILES:'):
            smiles = line.replace('SMILES:', '').strip()
            design['SMILES'] = smiles
        
        # Extract description
        if line.startswith('Description:'):
            desc = line.replace('Description:', '').strip()
            design['description'] = desc
        
        # Extract expected score
        if line.startswith('Expected Score:'):
            score = line.replace('Expected Score:', '').strip()
            design['expected_score'] = score
    
    if 'SMILES' in design and design['SMILES']:
        results.append(design)

print(f"\n✓ Parsed {len(results)} lipid designs")

# 保存到JSON
output_data = {
    'generation_date': '2025-12-11',
    'model': model_name,
    'task': 'Generate 10 high-efficiency ionizable lipid SMILES (score 9-10)',
    'total_designs': len(results),
    'designs': results
}

with open('result_AI.json', 'w') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print("\n" + "="*80)
print("Generation Complete!")
print("="*80)
print(f"Total designs generated: {len(results)}")
print("\nOutput files:")
print("  - result_AI.json (structured data)")
print("  - generation_raw_response.txt (raw model output)")
print("="*80)

# 显示生成的SMILES
if results:
    print("\nGenerated Ionizable Lipids:")
    for i, design in enumerate(results, 1):
        smiles = design.get('SMILES', 'N/A')
        score = design.get('expected_score', 'N/A')
        # 检查SMILES是否包含点号（表示多个分子）
        has_dot = '.' in smiles
        status = "⚠️ MULTIPLE MOLECULES" if has_dot else "✓"
        print(f"\n  {i}. {status}")
        print(f"     SMILES: {smiles[:100]}{'...' if len(smiles) > 100 else ''}")
        print(f"     Score: {score}")
else:
    print("\n⚠️  No valid designs were parsed. Check generation_raw_response.txt")
