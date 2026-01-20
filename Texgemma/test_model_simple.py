#!/usr/bin/env python3
"""
简单测试 txgemma-27b-chat 模型输出
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载一个测试分子
data_file = '/mnt/3fs/dots-pretrain/leshu/workspace/Texgemma/data/virtual_library_preprocessed.jsonl'
with open(data_file, 'r') as f:
    mol_data = json.loads(f.readline())

print("="*80)
print(f"Test Molecule ID: {mol_data['ID']}")
print(f"SMILES: {mol_data['SMILES']}")
print("="*80)

# 加载模型
print("\nLoading model...")
device = torch.device('cuda:0')

tokenizer = AutoTokenizer.from_pretrained(
    "google/txgemma-27b-chat",
    token=True,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    "google/txgemma-27b-chat",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    token=True,
    local_files_only=True
)

print("✓ Model loaded\n")

# 创建 prompt
prompt = f"""<start_of_turn>user
Predict the mRNA transfection efficiency (1-10 score) for this lipid molecule:

SMILES: {mol_data['SMILES']}

Features: {mol_data['tail_count']} tails, linker length {mol_data['linker_length']}

Provide:
Efficiency Score: [1-10]
Reason: [Brief explanation]<end_of_turn>
<start_of_turn>model
"""

print("PROMPT:")
print("-"*80)
print(prompt)
print("-"*80)

# Generate
print("\nGenerating response...")
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print("\nMODEL RESPONSE:")
print("="*80)
print(response)
print("="*80)
