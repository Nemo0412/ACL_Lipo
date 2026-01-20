#!/usr/bin/env python3
"""
使用TXGemma模型生成10个高效的ionizable lipid SMILES
基于已知的10分高效分子示例
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os

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
    trust_remote_code=True,
    load_in_8bit=False,
    load_in_4bit=False
)

print(f"✓ 模型加载完成")
print()

# 创建包含高分示例的prompt
prompt = """<start_of_turn>user
You are an expert in lipid nanoparticle design for mRNA delivery. I will show you 4 examples of EXCELLENT lipid molecules that achieve a transfection efficiency score of 10 out of 10. 

EXAMPLES OF HIGH-SCORING (10/10) LIPIDS:

1. SMILES: CCCCCCCC\\C=C/CCCCCCCCNC(=O)C(CCCCCOC(=O)CCC(C)CCCCC)NCCN1CCCC1
   Score: 10/10
   Key features: Long alkyl chain (C18), ester linkage, imidazoline ring, secondary amines

2. SMILES: CCCCCCCCCCC(=O)OCCCCCC(NC1=CNN=C1)C(=O)NCCCCCCCC\\C=C/CCCCCCCC
   Score: 10/10
   Key features: Imidazole ring, ester bond, amide linkages, C18 unsaturated chain

3. SMILES: CCCCCCCCCCCCCCCCNC(=O)C(CCCCCOC(=O)CCCCCCCCCC)NC1=CNN=C1
   Score: 10/10
   Key features: C16 saturated chain, imidazole ring, ester linkage, amide bonds

4. SMILES: CCCCCCCCCCCCCCCCCCNC(=O)C(CCCCCOC(=O)CCCCCCCCCC)NC1=CNN=C1
   Score: 10/10
   Key features: C18 saturated chain, imidazole ring, ester linkage, amide bonds

COMMON SUCCESS PATTERNS IN THESE MOLECULES:
- Imidazole or imidazoline rings (ionizable groups for endosomal escape)
- Long hydrophobic tails (C16-C18)
- Ester bonds for biodegradability
- Amide linkages for stability
- Secondary or tertiary amines
- Both saturated and unsaturated chains work well

YOUR TASK:
Design 10 NEW lipid molecules with similar structural features that should also achieve 9-10/10 transfection efficiency.

CRITICAL REQUIREMENTS:
1. Each SMILES must be a SINGLE MOLECULE (no dots ".", no salts, no multiple components)
2. Use similar structural motifs: imidazole/imidazoline rings, long alkyl chains (C14-C20), ester bonds
3. Vary the structures to explore chemical space (different chain lengths, branching, ring positions)
4. Must be synthetically feasible

OUTPUT FORMAT (JSON array):
[
  {
    "id": 1,
    "smiles": "YOUR_SINGLE_MOLECULE_SMILES_HERE",
    "name": "Descriptive name",
    "design_rationale": "Brief explanation of key features",
    "predicted_score": 9 or 10
  },
  ...
]

Generate 10 novel lipid SMILES now. Each must be ONE connected molecule without dots.<end_of_turn>
<start_of_turn>model
```json
[
"""

print("=" * 80)
print("生成10个高效ionizable lipid结构（基于高分示例）")
print("=" * 80)
print()

inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("正在生成... (这可能需要几分钟)")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=4096,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print("=" * 80)
print("模型响应:")
print("=" * 80)
print(response[:3000])
print()

# 尝试解析JSON
try:
    # 模型应该已经开始生成JSON了，因为prompt以 ```json\n[ 结尾
    # 查找JSON数组
    full_response = "```json\n[\n" + response
    
    start = full_response.find('[')
    end = full_response.rfind(']') + 1
    
    if start != -1 and end > start:
        json_str = full_response[start:end]
        
        # 尝试修复可能的JSON问题
        json_str = json_str.replace('```', '')  # 移除markdown代码块标记
        
        lipids = json.loads(json_str)
        
        print("=" * 80)
        print(f"成功解析 {len(lipids)} 个lipid结构")
        print("=" * 80)
        
        # 验证和清理SMILES
        valid_lipids = []
        for i, lipid in enumerate(lipids):
            smiles = lipid.get('smiles', '')
            
            # 检查是否包含点号（多个分子）
            if '.' in smiles:
                print(f"⚠ Lipid {i+1}: SMILES包含点号（多分子），跳过")
                continue
            
            # 检查是否为空
            if not smiles or len(smiles) < 20:
                print(f"⚠ Lipid {i+1}: SMILES太短或为空，跳过")
                continue
            
            valid_lipids.append(lipid)
            print(f"✓ Lipid {i+1}: {lipid.get('name', 'Unknown')}")
            print(f"  SMILES: {smiles[:100]}...")
            print(f"  Score: {lipid.get('predicted_score', 'N/A')}")
            print(f"  Rationale: {lipid.get('design_rationale', '')[:100]}...")
            print()
        
        if valid_lipids:
            # 保存到result_AI.json
            with open('result_AI.json', 'w') as f:
                json.dump(valid_lipids, f, indent=2, ensure_ascii=False)
            
            print("=" * 80)
            print(f"✓ 已保存 {len(valid_lipids)} 个有效的lipid结构到 result_AI.json")
            print("=" * 80)
            
            # 也保存完整响应
            with open('result_AI_raw.txt', 'w') as f:
                f.write(full_response)
            print("完整响应已保存到 result_AI_raw.txt")
        else:
            print("✗ 没有有效的SMILES结构")
            with open('result_AI_raw.txt', 'w') as f:
                f.write(full_response)
            print("原始响应已保存到 result_AI_raw.txt")
    else:
        print("✗ 未找到有效的JSON数组")
        with open('result_AI_raw.txt', 'w') as f:
            f.write(full_response)
        print("原始响应已保存到 result_AI_raw.txt")
        
except json.JSONDecodeError as e:
    print(f"✗ JSON解析失败: {e}")
    with open('result_AI_raw.txt', 'w') as f:
        f.write(full_response if 'full_response' in locals() else response)
    print("原始响应已保存到 result_AI_raw.txt")
    print("\n尝试手动查找SMILES...")
    
    # 尝试手动提取SMILES
    import re
    smiles_pattern = r'"smiles":\s*"([^"]+)"'
    matches = re.findall(smiles_pattern, response)
    
    if matches:
        print(f"找到 {len(matches)} 个SMILES:")
        for i, smiles in enumerate(matches[:10], 1):
            if '.' not in smiles and len(smiles) > 20:
                print(f"  {i}. {smiles[:80]}...")
                
except Exception as e:
    print(f"✗ 处理失败: {e}")
    import traceback
    traceback.print_exc()
    with open('result_AI_raw.txt', 'w') as f:
        f.write(full_response if 'full_response' in locals() else response)
    print("原始响应已保存到 result_AI_raw.txt")

print("\n完成!")
