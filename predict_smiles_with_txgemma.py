#!/usr/bin/env python3
"""
使用TxGemma-27B模型预测SMILES的mRNA转染效率
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# 配置
MODEL_PATH = "/mnt/3fs/dots-pretrain/leshu/workspace/Txgemma/txgemma-27b-chat"
EXCEL_FILE = "/mnt/3fs/dots-pretrain/leshu/workspace/Smiles比较.xlsx"

def load_model_and_tokenizer(model_path):
    """加载模型和tokenizer"""
    print(f"正在加载模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print("模型加载完成！")
    return model, tokenizer

def create_prompt(smiles):
    """创建提示词"""
    prompt = f"""You are an expert in lipid nanoparticle (LNP) design and mRNA delivery. 

Please analyze the following ionizable lipid SMILES structure and predict its mRNA transfection efficiency:

SMILES: {smiles}

Please provide:
1. Predicted mRNA transfection efficiency (qualitative assessment: High/Medium/Low and quantitative if possible)
2. Detailed structural analysis including:
   - Hydrophobic tail length and its impact
   - Ester bond type and positioning
   - Number of lipid tails
   - Number of tertiary amines
   - Head group structure
3. Rationale for your prediction, explaining how each structural feature influences:
   - Membrane fusion capability
   - Endosomal escape efficiency
   - Biodegradability
   - Overall delivery performance

Please provide a comprehensive analysis with specific attention to structure-activity relationships."""
    
    return prompt

def predict_transfection_efficiency(model, tokenizer, smiles, smiles_name):
    """预测单个SMILES的转染效率"""
    print(f"\n{'='*80}")
    print(f"分析 {smiles_name}")
    print(f"{'='*80}")
    
    prompt = create_prompt(smiles)
    
    # 格式化为对话格式
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 应用聊天模板
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 编码输入
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # 生成回复
    print("正在生成预测...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取模型的回复（去除输入部分）
    if "<start_of_turn>model" in response:
        response = response.split("<start_of_turn>model")[-1].strip()
    elif formatted_prompt in response:
        response = response.replace(formatted_prompt, "").strip()
    
    print(f"\n预测结果:")
    print(response)
    
    return response

def main():
    # 读取Excel文件
    print(f"读取Excel文件: {EXCEL_FILE}")
    df = pd.read_excel(EXCEL_FILE, header=None)  # 不使用第一行作为列名
    
    print(f"\n找到 {len(df)} 个SMILES结构")
    print(df)
    
    # 解析数据：第一列是序号，第二列是名称，第三列是SMILES
    # 将数据重新组织
    smiles_data = []
    for idx, row in df.iterrows():
        if len(row) >= 3:
            # 第二列是名称，第三列是SMILES
            name = str(row[1]) if pd.notna(row[1]) else f'SMILES_{idx+1}'
            smiles = str(row[2]) if pd.notna(row[2]) else ''
            if smiles and smiles != 'nan':
                smiles_data.append({'Name': name, 'SMILES': smiles})
    
    print(f"\n解析到 {len(smiles_data)} 个有效的SMILES结构")
    for data in smiles_data:
        print(f"  - {data['Name']}: {data['SMILES'][:50]}...")
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    
    # 对每个SMILES进行预测
    results = []
    for data in smiles_data:
        smiles_name = data['Name']
        smiles = data['SMILES']
        
        if not smiles:
            print(f"警告: {smiles_name} 没有SMILES结构")
            continue
        
        prediction = predict_transfection_efficiency(model, tokenizer, smiles, smiles_name)
        
        results.append({
            'Name': smiles_name,
            'SMILES': smiles,
            'Prediction': prediction
        })
    
    # 保存结果
    output_file = "/mnt/3fs/dots-pretrain/leshu/workspace/txgemma_predictions.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("TxGemma-27B mRNA转染效率预测结果\n")
        f.write("=" * 100 + "\n\n")
        
        for result in results:
            f.write(f"\n{'='*100}\n")
            f.write(f"名称: {result['Name']}\n")
            f.write(f"SMILES: {result['SMILES']}\n")
            f.write(f"{'-'*100}\n")
            f.write(f"预测结果:\n{result['Prediction']}\n")
    
    print(f"\n结果已保存到: {output_file}")
    
    # 也保存为Excel格式
    results_df = pd.DataFrame(results)
    excel_output = "/mnt/3fs/dots-pretrain/leshu/workspace/txgemma_predictions.xlsx"
    results_df.to_excel(excel_output, index=False)
    print(f"Excel结果已保存到: {excel_output}")

if __name__ == "__main__":
    main()
