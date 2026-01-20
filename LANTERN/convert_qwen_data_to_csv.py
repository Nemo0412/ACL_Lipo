"""
Convert Haiyu-Qwen30B JSONL data to CSV format for LANTERN evaluation
"""
import json
import pandas as pd
import re
import os

def extract_smiles_and_label_from_jsonl(jsonl_file, task='efficiency'):
    """
    Extract SMILES and labels from JSONL file
    
    Args:
        jsonl_file: path to JSONL file
        task: 'efficiency' or 'toxicity'
    
    Returns:
        DataFrame with SMILES and Target columns
    """
    data = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            messages = item['messages']
            
            # Extract SMILES from user message
            user_msg = messages[1]['content']
            
            # Try different patterns for SMILES extraction
            if task == 'efficiency':
                smiles_match = re.search(r'molecular structure: ([^\?]+)\?', user_msg)
            else:  # toxicity
                smiles_match = re.search(r'toxic\?\s+([A-Za-z0-9@\[\]\(\)=#\\/\-\+]+)\s+Molecular', user_msg)
            
            if not smiles_match:
                continue
            
            smiles = smiles_match.group(1).strip()
            
            # Extract label from assistant message
            assistant_msg = messages[2]['content']
            
            if task == 'efficiency':
                # Extract efficiency score (1-10)
                score_match = re.search(r'is (\d+)\.', assistant_msg)
                if score_match:
                    label = int(score_match.group(1))
                    data.append({'SMILES': smiles, 'Target': label})
            else:  # toxicity
                # Extract toxicity value (0 or 1)
                if 'Toxicity value: 0' in assistant_msg:
                    label = 0
                elif 'Toxicity value: 1' in assistant_msg:
                    label = 1
                else:
                    continue
                data.append({'SMILES': smiles, 'Target': label})
    
    return pd.DataFrame(data)


def main():
    # Paths
    base_path = '/mnt/3fs/dots-pretrain/leshu/workspace/Haiyu-Qwen30B/data'
    efficiency_file = os.path.join(base_path, 'efficiency_test_data_rdkit.jsonl')
    toxicity_file = os.path.join(base_path, 'toxic_data/toxicity_test_data_rdkit.jsonl')
    
    output_dir = '/mnt/3fs/dots-pretrain/leshu/workspace/LANTERN/LANTERN/data'
    
    # Convert efficiency data
    print("Converting efficiency test data...")
    eff_df = extract_smiles_and_label_from_jsonl(efficiency_file, task='efficiency')
    print(f"Extracted {len(eff_df)} efficiency samples")
    print(f"Efficiency label distribution:\n{eff_df['Target'].value_counts().sort_index()}")
    
    eff_output = os.path.join(output_dir, 'efficiency_test_qwen.csv')
    eff_df.to_csv(eff_output, index=False)
    print(f"Saved to: {eff_output}\n")
    
    # Convert toxicity data
    print("Converting toxicity test data...")
    tox_df = extract_smiles_and_label_from_jsonl(toxicity_file, task='toxicity')
    print(f"Extracted {len(tox_df)} toxicity samples")
    print(f"Toxicity label distribution:\n{tox_df['Target'].value_counts().sort_index()}")
    
    tox_output = os.path.join(output_dir, 'toxicity_test_qwen.csv')
    tox_df.to_csv(tox_output, index=False)
    print(f"Saved to: {tox_output}\n")
    
    print("=" * 80)
    print("Conversion complete!")
    print(f"Efficiency test: {len(eff_df)} samples -> {eff_output}")
    print(f"Toxicity test: {len(tox_df)} samples -> {tox_output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
