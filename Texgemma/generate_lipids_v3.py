#!/usr/bin/env python3
"""
使用TXGemma模型生成10个高效的ionizable lipid SMILES
要求：单个分子、可合成、预测效率9-10分
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
    trust_remote_code=True
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

# 创建详细的prompt
prompt = """<start_of_turn>user
You are an expert in lipid nanoparticle design for mRNA delivery. I need you to design 10 novel ionizable lipid molecules with excellent mRNA transfection efficiency.

CRITICAL REQUIREMENTS:
1. Each lipid must be a SINGLE molecule (not a salt, not multiple components)
2. SMILES must represent only ONE neutral molecule (no "." separators, no ions like Cl- or Na+)
3. The lipid should be synthesizable in a standard chemistry lab
4. Expected transfection efficiency: 9-10 out of 10

DESIGN PRINCIPLES for high efficiency:
- Include ionizable amine groups (tertiary or quaternary amines) for endosomal escape
- Use biodegradable ester linkages (like DLin-MC3-DMA, SM-102)
- Optimal hydrophobic tail length: C12-C18
- Include 2-4 lipid tails for proper nanoparticle formation
- Consider branched structures for improved efficacy
- Use amino acid-based head groups (like lysine derivatives)

OUTPUT FORMAT (JSON):
For each of the 10 lipids, provide:
{
  "id": 1,
  "smiles": "[SINGLE MOLECULE SMILES - no dots, no salts]",
  "name": "Descriptive name",
  "design_rationale": "Why this structure should have 9-10 efficiency",
  "predicted_score": 9 or 10
}

EXAMPLES of GOOD ionizable lipids:
- DLin-MC3-DMA: CCCCCCCC/C=C\\CCCCCCCC(=O)OCC(COC(=O)CCCCCCC/C=C\\CCCCCCCC)OC(=O)CCCCCCC/C=C\\CCCCCCCC
- SM-102 has ester bonds and tertiary amines
- ALC-0315 (Pfizer vaccine lipid)

Please generate 10 novel lipid SMILES now. Make sure each SMILES is:
- A single continuous molecule
- Contains NO dots (.)
- Contains NO salts or counter-ions
- Can be drawn as one connected structure
- Focuses on ionizable amino lipids with biodegradable linkers

Respond ONLY with a valid JSON array of 10 lipid structures.<end_of_turn>
<start_of_turn>model
"""

print("=" * 80)
print("生成10个高效ionizable lipid结构")
print("=" * 80)
print()

inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("正在生成... (这可能需要几分钟)")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=4096,  # 需要足够长以生成10个分子
        temperature=0.8,      # 稍高温度增加创新性
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print("=" * 80)
print("模型响应:")
print("=" * 80)
print(response[:2000])  # 打印前2000字符
print()

# 尝试解析JSON
try:
    # 查找JSON数组
    start = response.find('[')
    end = response.rfind(']') + 1
    
    if start != -1 and end > start:
        json_str = response[start:end]
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
                print(f"⚠ Lipid {i+1}: SMILES包含点号，跳过")
                continue
            
            # 检查是否为空
            if not smiles or len(smiles) < 10:
                print(f"⚠ Lipid {i+1}: SMILES太短或为空，跳过")
                continue
            
            valid_lipids.append(lipid)
            print(f"✓ Lipid {i+1}: {lipid.get('name', 'Unknown')}")
            print(f"  SMILES: {smiles[:80]}...")
            print(f"  Score: {lipid.get('predicted_score', 'N/A')}")
            print()
        
        if valid_lipids:
            # 保存到result_AI.json
            with open('result_AI.json', 'w') as f:
                json.dump(valid_lipids, f, indent=2, ensure_ascii=False)
            
            print("=" * 80)
            print(f"✓ 已保存 {len(valid_lipids)} 个有效的lipid结构到 result_AI.json")
            print("=" * 80)
        else:
            print("✗ 没有有效的SMILES结构")
            # 保存原始响应以便调试
            with open('result_AI_raw.txt', 'w') as f:
                f.write(response)
            print("原始响应已保存到 result_AI_raw.txt")
    else:
        print("✗ 未找到有效的JSON数组")
        with open('result_AI_raw.txt', 'w') as f:
            f.write(response)
        print("原始响应已保存到 result_AI_raw.txt")
        
except json.JSONDecodeError as e:
    print(f"✗ JSON解析失败: {e}")
    with open('result_AI_raw.txt', 'w') as f:
        f.write(response)
    print("原始响应已保存到 result_AI_raw.txt")
except Exception as e:
    print(f"✗ 处理失败: {e}")
    with open('result_AI_raw.txt', 'w') as f:
        f.write(response)
    print("原始响应已保存到 result_AI_raw.txt")

print("\n完成!")
