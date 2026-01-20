#!/usr/bin/env python3
"""
使用TXGemma模型生成10个高效的lipid SMILES（目标9-10分）
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os

# 设置离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print("="*80)
print("使用TXGemma生成高效lipid SMILES")
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
You are an expert in lipid nanoparticle design for mRNA delivery. Based on your knowledge of high-performing ionizable lipids, please design 10 novel lipid molecules with excellent mRNA transfection efficiency (score 9-10 out of 10).

For each lipid, provide:
1. A valid SMILES string
2. Key structural features (amino acid base, protection group, linker length, number of tails, ester bonds)
3. Brief explanation of why this design should achieve high transfection efficiency

Guidelines for high-performing lipids:
- Ionizable amino head groups (tertiary amines, pKa ~6-7)
- Biodegradable ester or disulfide linkages
- Optimal hydrophobic tail length (C12-C18)
- Multiple lipid tails (2-4 for balance)
- Branched or asymmetric structures often perform better
- Consider successful examples like DLin-MC3-DMA, ALC-0315, SM-102

Please provide exactly 10 lipid designs in this format for each:

Design [Number]:
SMILES: [valid SMILES string]
Amino acid: [e.g., K, KK, Histidine]
Protection group: [e.g., None, Acetyl, Boc]
Linker length: [number] carbons
Tail count: [number]
Ester bonds (OCOO): [number]
Predicted Score: 9-10
Rationale: [2-3 sentences explaining the design choices and expected high performance]

<end_of_turn>
<start_of_turn>model
"""

print("\n" + "="*80)
print("Generating 10 high-efficiency lipid designs...")
print("="*80 + "\n")

# Generate
inputs = tokenizer(generation_prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=3000,
        temperature=0.8,
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

# 尝试解析响应并创建结构化输出
results = []

# 简单的解析逻辑
lines = response.split('\n')
current_design = {}

for line in lines:
    line = line.strip()
    
    if line.startswith('Design') and ':' in line:
        if current_design:
            results.append(current_design)
        current_design = {'ID': len(results) + 1}
    
    elif line.startswith('SMILES:'):
        current_design['SMILES'] = line.replace('SMILES:', '').strip()
    
    elif line.startswith('Amino acid:'):
        current_design['amino_acid'] = line.replace('Amino acid:', '').strip()
    
    elif line.startswith('Protection group:'):
        current_design['protection_group'] = line.replace('Protection group:', '').strip()
    
    elif line.startswith('Linker length:'):
        current_design['linker_length'] = line.replace('Linker length:', '').replace('carbons', '').strip()
    
    elif line.startswith('Tail count:'):
        current_design['tail_count'] = line.replace('Tail count:', '').strip()
    
    elif line.startswith('Ester bonds'):
        current_design['ester_bonds_OCOO'] = line.split(':')[-1].strip()
    
    elif line.startswith('Predicted Score:'):
        current_design['predicted_score'] = line.replace('Predicted Score:', '').strip()
    
    elif line.startswith('Rationale:'):
        current_design['rationale'] = line.replace('Rationale:', '').strip()

# 添加最后一个设计
if current_design and 'SMILES' in current_design:
    results.append(current_design)

print(f"\n✓ Parsed {len(results)} lipid designs")

# 保存到JSON
output_data = {
    'generation_date': '2025-12-11',
    'model': model_name,
    'prompt': 'Generate 10 high-efficiency lipid SMILES (score 9-10)',
    'total_designs': len(results),
    'designs': results
}

with open('result_AI.json', 'w') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print("\n" + "="*80)
print("Generation Complete!")
print("="*80)
print(f"Total designs generated: {len(results)}")
print("Output files:")
print("  - result_AI.json (structured data)")
print("  - generation_raw_response.txt (raw model output)")
print("="*80)

# 显示生成的SMILES
if results:
    print("\nGenerated SMILES:")
    for i, design in enumerate(results, 1):
        smiles = design.get('SMILES', 'N/A')
        score = design.get('predicted_score', 'N/A')
        print(f"  {i}. {smiles[:80]}... (Score: {score})")
