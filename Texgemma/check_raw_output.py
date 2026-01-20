#!/usr/bin/env python3
"""
检查模型的原始输出，用于调试
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def create_prediction_prompt(mol_data):
    """创建预测 prompt"""
    user_prompt = f"""Please predict the mRNA transfection efficiency for the following lipid molecule.

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

Format your response as:
Efficiency Score: [1-10]
Reason: [Your detailed rationale]"""

    return user_prompt


def main():
    # 加载数据
    data_file = '/mnt/3fs/dots-pretrain/leshu/workspace/Texgemma/data/virtual_library_preprocessed.jsonl'
    
    molecules = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            molecules.append(json.loads(line))
            if i >= 2:  # 只测试前3个
                break
    
    print(f"Loaded {len(molecules)} test molecules\n")
    
    # 加载模型
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    model_name = "google/txgemma-27b-predict"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=True
    )
    
    print("✓ Model loaded\n")
    
    # 测试预测
    for mol_data in molecules:
        print("="*100)
        print(f"Testing Molecule ID: {mol_data['ID']}")
        print("="*100)
        
        prompt = create_prediction_prompt(mol_data)
        print("\n📝 PROMPT:")
        print("-"*100)
        print(prompt)
        print("-"*100)
        
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
        
        print("\n🤖 RAW MODEL RESPONSE:")
        print("-"*100)
        print(response)
        print("-"*100)
        print("\n")


if __name__ == '__main__':
    main()
