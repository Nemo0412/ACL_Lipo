#!/usr/bin/env python3
"""
数据预处理：将 Excel 虚拟分子库转换为模型友好的格式
"""

import pandas as pd
import json
import os

def preprocess_virtual_library(excel_file, output_jsonl):
    """
    预处理虚拟分子库数据
    
    输入: Excel 文件，包含以下列：
    - Number: 分子编号
    - Amino acid: 氨基酸类型
    - Protection: 保护基团
    - linker: linker 长度
    - OCOO: 酯键数量
    - tail: 尾链数量  
    - smiles: SMILES 结构
    
    输出: JSONL 文件，每行一个分子，包含结构化信息
    """
    
    print(f"Loading data from: {excel_file}")
    df = pd.read_excel(excel_file)
    
    print(f"Total molecules: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    processed_data = []
    
    for idx, row in df.iterrows():
        # 提取基本信息
        mol_data = {
            'ID': int(row['Number']),
            'SMILES': row['smiles'],
            'amino_acid': row['Amino acid'],
            'protection_group': row['Protection'],
            'linker_length': int(row['linker']),
            'ester_bonds_OCOO': int(row['OCOO']),
            'tail_count': int(row['tail'])
        }
        
        # 创建特征描述（用于 prompt）
        features_desc = f"{mol_data['amino_acid']}-based lipid with {mol_data['protection_group']} protection, "
        features_desc += f"linker length {mol_data['linker_length']}, "
        features_desc += f"{'with' if mol_data['ester_bonds_OCOO'] > 0 else 'without'} OCOO ester bonds, "
        features_desc += f"{mol_data['tail_count']} hydrophobic tail{'s' if mol_data['tail_count'] > 1 else ''}"
        
        mol_data['features_description'] = features_desc
        
        # 分析尾链长度（从 SMILES 推断）
        # 简单启发式：统计碳链长度
        smiles = mol_data['SMILES']
        c_count = smiles.count('C')
        mol_data['estimated_total_carbons'] = c_count
        
        processed_data.append(mol_data)
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1} molecules...")
    
    # 保存为 JSONL
    print(f"\nSaving to: {output_jsonl}")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for mol in processed_data:
            f.write(json.dumps(mol, ensure_ascii=False) + '\n')
    
    print(f"✓ Saved {len(processed_data)} molecules")
    
    # 统计信息
    print("\n" + "="*80)
    print("Data Statistics")
    print("="*80)
    
    df_processed = pd.DataFrame(processed_data)
    
    print(f"\nAmino acid distribution:")
    print(df_processed['amino_acid'].value_counts())
    
    print(f"\nProtection group distribution:")
    print(df_processed['protection_group'].value_counts())
    
    print(f"\nLinker length distribution:")
    print(df_processed['linker_length'].value_counts().sort_index())
    
    print(f"\nEster bonds (OCOO) distribution:")
    print(df_processed['ester_bonds_OCOO'].value_counts().sort_index())
    
    print(f"\nTail count distribution:")
    print(df_processed['tail_count'].value_counts().sort_index())
    
    # 保存统计信息
    stats_file = output_jsonl.replace('.jsonl', '_stats.txt')
    with open(stats_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Virtual Library Statistics\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total molecules: {len(processed_data)}\n\n")
        f.write(f"Amino acid distribution:\n{df_processed['amino_acid'].value_counts()}\n\n")
        f.write(f"Protection group distribution:\n{df_processed['protection_group'].value_counts()}\n\n")
        f.write(f"Linker length distribution:\n{df_processed['linker_length'].value_counts().sort_index()}\n\n")
        f.write(f"Ester bonds (OCOO) distribution:\n{df_processed['ester_bonds_OCOO'].value_counts().sort_index()}\n\n")
        f.write(f"Tail count distribution:\n{df_processed['tail_count'].value_counts().sort_index()}\n")
    
    print(f"\n✓ Statistics saved to: {stats_file}")
    
    # 显示几个示例
    print("\n" + "="*80)
    print("Sample Molecules")
    print("="*80)
    
    for i in range(min(3, len(processed_data))):
        mol = processed_data[i]
        print(f"\nMolecule {i+1} (ID={mol['ID']}):")
        print(f"  SMILES: {mol['SMILES'][:80]}...")
        print(f"  Features: {mol['features_description']}")
        print(f"  Carbons: ~{mol['estimated_total_carbons']}")
    
    return processed_data


def main():
    input_file = '/mnt/3fs/dots-pretrain/leshu/workspace/Txgemma/data/10000-virtual library.xlsx'
    output_file = '/mnt/3fs/dots-pretrain/leshu/workspace/Texgemma/data/virtual_library_preprocessed.jsonl'
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    processed_data = preprocess_virtual_library(input_file, output_file)
    
    print("\n" + "="*80)
    print("Preprocessing Complete!")
    print("="*80)
    print(f"Output file: {output_file}")
    print(f"Ready for model prediction.")


if __name__ == '__main__':
    main()
