#!/usr/bin/env python3
"""
TxGemma-27B-Chat verify pass: two-stage scoring with incremental JSON saves.

NOTE: An embedding-space variant (no decode/re-encode between passes) is
available at txgemma/embedding_agent.py. It passes Pass-1 hidden states
directly as inputs_embeds to Pass 2, removing the tokenisation bottleneck.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from tqdm import tqdm
import os
from datetime import datetime

def load_preprocessed_data(jsonl_file):
    """Load preprocessed JSONL records."""
    print(f"Loading preprocessed data from: {jsonl_file}")
    
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"✓ Loaded {len(data)} molecules")
    return data


def create_prediction_prompt(mol_data, need_detailed_reason=False):
    """Build prompt for fast score-only pass or detailed rationale pass.

    Args:
        mol_data: molecule dict from preprocess.
        need_detailed_reason: if True, ask for long-form rationale.
    """
    
    if need_detailed_reason:
        # Long prompt for high/low scorers
        conversation = f"""<start_of_turn>user
You are an expert in lipid nanoparticles and mRNA delivery. Please predict the mRNA transfection efficiency for the following lipid molecule.

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

Please respond in this exact format:
Efficiency Score: [number from 1-10]
Reason: [Your detailed analysis of the molecular structure and its impact on transfection efficiency]<end_of_turn>
<start_of_turn>model
"""
    else:
        # Short prompt: score only (fast path)
        conversation = f"""<start_of_turn>user
You are an expert in lipid nanoparticles and mRNA delivery. Predict the transfection efficiency score (1-10) for this molecule:

SMILES: {mol_data['SMILES']}
- Amino acid: {mol_data['amino_acid']}, Protection: {mol_data['protection_group']}
- Linker: {mol_data['linker_length']}C, Tails: {mol_data['tail_count']}, OCOO: {mol_data['ester_bonds_OCOO']}

Respond ONLY with: Efficiency Score: [number]<end_of_turn>
<start_of_turn>model
"""
    
    return conversation


def extract_score_and_reason(response):
    """Parse score and rationale from free-form model text."""
    score = None
    reason = ""
    
    # Typical output: "## Efficiency Score: 3\n\n## Reason:\n..."
    # Score regexes (markdown-friendly)
    score_patterns = [
        r'##\s*[Ee]fficiency\s+[Ss]core\s*[:：]\s*(\d+)',  # markdown heading style
        r'[Ee]fficiency\s+[Ss]core\s*[:：]\s*(\d+)',
        r'##\s*[Ss]core\s*[:：]\s*(\d+)',
        r'[Ss]core\s*[:：]\s*(\d+)',
        r'rating\s+of\s+(\d+)',
        r'score\s+of\s+(\d+)',
    ]
    
    for pattern in score_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 10:
                break
    
    # Fallback: first 1..10 token
    if score is None:
        numbers = re.findall(r'\b([1-9]|10)\b', response)
        if numbers:
            score = int(numbers[0])
    
    # Rationale regexes (markdown-friendly)
    reason_patterns = [
        r'##\s*[Rr]eason\s*[:：]?\s*\n+(.+)',  # markdown heading style
        r'[Rr]eason\s*[:：]\s*(.+)',
        r'##\s*[Aa]nalysis\s*[:：]?\s*\n+(.+)',
        r'[Rr]ationale\s*[:：]\s*(.+)',
    ]
    
    for pattern in reason_patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            reason = match.group(1).strip()
            break
    
    # If rationale missing/short, strip score heading lines from full text
        lines = response.split('\n')
        reason_lines = []
        skip_next = False
        for line in lines:
            if re.search(r'##\s*[Ee]fficiency\s+[Ss]core', line, re.IGNORECASE):
                skip_next = True
                continue
            if skip_next and line.strip() == '':
                skip_next = False
                continue
            reason_lines.append(line)
        reason = '\n'.join(reason_lines).strip()
    
    # Default score if parsing fails
    if score is None:
        score = 5
        reason = f"[WARNING: No score found, using 5]\n\n{response[:500]}"
    
    # Last resort: keep entire response as rationale
    if len(reason) < 20:
        reason = response.strip()
    
    return score, reason


def save_results_incremental(results, output_file):
    """Write current results to disk (sorted by score)."""
    results_sorted = sorted(results, key=lambda x: x['efficiency_score'], reverse=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_sorted, f, indent=2, ensure_ascii=False)


def predict_batch(model, tokenizer, molecules, device='cuda', output_file='result.json'):
    """Two-pass prediction with resume support and periodic saves."""
    model.eval()
    results = []
    
    # Resume from partial output if present
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                if existing_results:
                    results = existing_results
                    completed_ids = {r['ID'] for r in results}
                    molecules = [m for m in molecules if m['ID'] not in completed_ids]
                    print(f"✓ Resuming: {len(results)} already completed, {len(molecules)} remaining")
        except:
            pass
    
    print(f"\n{'='*100}")
    print(f"Starting Prediction for {len(molecules)} molecules")
    print(f"{'='*100}\n")
    
    debug_count = 0  # verbose dumps for first few detailed passes
    detailed_count = 0  # how many long rationales were generated
    
    for i, mol_data in enumerate(tqdm(molecules, desc="Predicting")):
        mol_id = mol_data['ID']
        
        try:
            # Pass 1: quick score (short generation budget)
            prompt_quick = create_prediction_prompt(mol_data, need_detailed_reason=False)
            
            # Tokenize
            inputs = tokenizer(prompt_quick, return_tensors="pt", truncation=True, max_length=2048).to(device)
            
            # Generate (fast path)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode
            response_quick = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Parse score from quick response
            score, _ = extract_score_and_reason(response_quick)
            
            # Pass 2: optional long rationale for extreme scores
            reason = ""
            if score >= 7 or score == 1:
                detailed_count += 1
                # Regenerate with detailed instructions
                prompt_detailed = create_prediction_prompt(mol_data, need_detailed_reason=True)
                inputs_detailed = tokenizer(prompt_detailed, return_tensors="pt", truncation=True, max_length=2048).to(device)
                
                with torch.no_grad():
                    outputs_detailed = model.generate(
                        **inputs_detailed,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response_detailed = tokenizer.decode(outputs_detailed[0][inputs_detailed['input_ids'].shape[1]:], skip_special_tokens=True)
                score, reason = extract_score_and_reason(response_detailed)
                
                # Debug trace for first few detailed generations
                if debug_count < 5:
                    print(f"\n{'='*80}")
                    print(f"DEBUG - Molecule ID {mol_id} (Score={score}, Need detailed reason)")
                    print(f"{'='*80}")
                    print(f"Raw Response:\n{response_detailed[:800]}")
                    print(f"{'='*80}\n")
                    debug_count += 1
            else:
                # Mid scores: skip rationale to save compute
                reason = ""
            
            result = {
                'ID': mol_id,
                'SMILES': mol_data['SMILES'],
                'efficiency_score': score,
                'reason': reason
            }
            
            results.append(result)
            
            # Checkpoint every 10 molecules
            if len(results) % 10 == 0:
                save_results_incremental(results, output_file)
            
            # Progress log every 50 molecules
            if len(results) % 50 == 0:
                print(f"\n[{len(results)}/{len(molecules) + len(results)}] Processed {len(results)} molecules")
                print(f"  Last: ID={mol_id}, Score={score}, Reason={'Yes' if reason else 'No'}")
                print(f"  Detailed reasons generated: {detailed_count}/{len(results)} ({100*detailed_count/len(results):.1f}%)")
                
                # Histogram for last 50 rows
                recent_scores = [r['efficiency_score'] for r in results[-50:]]
                score_dist = {s: recent_scores.count(s) for s in set(recent_scores)}
                print(f"  Recent 50 score distribution: {dict(sorted(score_dist.items()))}")
        
        except Exception as e:
            print(f"\n✗ Error processing molecule {mol_id}: {e}")
            result = {
                'ID': mol_id,
                'SMILES': mol_data['SMILES'],
                'efficiency_score': 5,
                'reason': f"ERROR: {str(e)}"
            }
            results.append(result)
    
    # Final flush to disk
    save_results_incremental(results, output_file)
    
    return results


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(base_dir, 'data', 'virtual_library_preprocessed.jsonl')
    output_file = os.path.join(base_dir, 'data', 'txgemma_verify_results.json')

    # Validate input path
    if not os.path.exists(data_file):
        print(f"✗ Data file not found: {data_file}")
        return
    
    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load molecules
    print("\n" + "="*100)
    print("Loading Data")
    print("="*100)
    molecules = load_preprocessed_data(data_file)
    
    # Load model from local cache only
    print("\n" + "="*100)
    print("Loading TXGemma-27B-Chat Model")
    print("="*100)
    
    local_model_path = os.path.join(base_dir, 'Txgemma', 'txgemma-27b-chat')
    model_name = local_model_path if os.path.exists(local_model_path) else "google/txgemma-27b-chat"
    print("Note: Using locally cached model (offline mode)")
    
    # Force offline Hub access
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    
    print(f"✓ Successfully loaded: {model_name}")
    
    # Run verify loop
    print("\n" + "="*100)
    print("Starting Predictions")
    print("="*100)
    print(f"Model: {model_name}")
    print(f"Total molecules: {len(molecules)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results will be saved incrementally to: {output_file}")
    
    start_time = datetime.now()
    
    results = predict_batch(
        model=model,
        tokenizer=tokenizer,
        molecules=molecules,
        device=device,
        output_file=output_file
    )
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print(f"\n✓ Predictions completed!")
    print(f"  Total time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
    print(f"  Average: {elapsed/len(molecules):.2f} seconds/molecule")
    
    # Score histogram
    score_dist = {}
    for r in results:
        score = r['efficiency_score']
        score_dist[score] = score_dist.get(score, 0) + 1
    
    print("\nScore Distribution:")
    for score in sorted(score_dist.keys(), reverse=True):
        count = score_dist[score]
        percentage = count / len(results) * 100
        bar = '█' * int(percentage / 2)
        print(f"  Score {score:2d}: {count:4d} molecules ({percentage:5.1f}%) {bar}")
    
    print(f"\n✓ Final results saved to: {output_file}")
    print("\n" + "="*100)
    print("Prediction Complete!")
    print("="*100)


if __name__ == '__main__':
    main()
