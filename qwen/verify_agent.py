"""
Descriptor-based efficiency heuristic (RDKit) + Qwen rationale generation.
"""
import os
import json
import time
from collections import Counter

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 70)
print("Verify agent: RDKit descriptors + Qwen rationale")
print("=" * 70)

# RDKit is required for descriptor features
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, Crippen

    print("RDKit OK")
except ImportError:
    print("RDKit is required. Install with: pip install rdkit")
    raise SystemExit(1)

# Optional subset for debugging
TEST_MODE = False
TEST_SIZE = 10
MODEL_NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-8B")

print(f"Test mode: {'on' if TEST_MODE else 'off'}")
if TEST_MODE:
    print(f"Test size: {TEST_SIZE}")
print("=" * 70)

# Load Qwen (offline cache if HF_HUB_OFFLINE=1)
print("\nLoading Qwen...")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
print("Qwen loaded.")


def calculate_molecular_descriptors(smiles: str) -> dict | None:
    """Return rounded RDKit descriptors or None if SMILES is invalid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        return {
            "molecular_weight": round(Descriptors.MolWt(mol), 1),
            "logP": round(Crippen.MolLogP(mol), 1),
            "hbd": Lipinski.NumHDonors(mol),
            "hba": Lipinski.NumHAcceptors(mol),
            "tpsa": round(Descriptors.TPSA(mol), 1),
            "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        }
    except Exception as e:
        print(f"Descriptor error: {e}")
        return None


def score_molecular_weight(mw: float) -> float:
    """MW subscore (ideal band ~160-480 Da)."""
    if 160 <= mw <= 480:
        return 2.0
    if 480 < mw <= 500:
        return 1.5
    if 140 <= mw < 160 or 500 < mw <= 520:
        return 1.0
    if 520 < mw <= 600:
        return 0.5
    return 0.0


def score_lipophilicity(logP: float) -> float:
    """logP subscore (ideal ~0-5)."""
    if 0 <= logP <= 3:
        return 2.0
    if 3 < logP <= 5:
        return 1.5
    if -0.4 <= logP < 0:
        return 1.0
    if 5 < logP <= 6:
        return 0.5
    return 0.0


def score_hbond(hbd: int, hba: int) -> float:
    """H-bond donor/acceptor subscore."""
    score = 0.0
    if hbd <= 5:
        score += 1.0
    elif hbd <= 7:
        score += 0.5

    if hba <= 10:
        score += 1.0
    elif hba <= 12:
        score += 0.5

    return score


def score_tpsa(tpsa: float) -> float:
    """TPSA subscore (ideal <140)."""
    if tpsa <= 140:
        return 2.0
    if 140 < tpsa <= 160:
        return 1.0
    if 160 < tpsa <= 180:
        return 0.5
    return 0.0


def score_flexibility(rotatable: int) -> float:
    """Rotatable bond subscore (ideal <10)."""
    if rotatable <= 10:
        return 2.0
    if 10 < rotatable <= 15:
        return 1.0
    if 15 < rotatable <= 20:
        return 0.5
    return 0.0


def calculate_efficiency_score(descriptors: dict) -> float:
    """Heuristic 0-10 score from Lipinski-style features (one decimal)."""
    if descriptors is None:
        return 0.0

    score = 0.0
    score += score_molecular_weight(descriptors["molecular_weight"])
    score += score_lipophilicity(descriptors["logP"])
    score += score_hbond(descriptors["hbd"], descriptors["hba"])
    score += score_tpsa(descriptors["tpsa"])
    score += score_flexibility(descriptors["rotatable_bonds"])

    # Shift and clip
    final_score = round(score + 2, 1)
    return min(final_score, 10.0)


def generate_reasoning(smiles: str, descriptors: dict, score: float) -> str:
    """Ask Qwen for an English narrative aligned with the heuristic score."""

    prompt = f"""You are a medicinal chemistry expert. Explain the efficiency score of this molecule in English.

Molecule SMILES: {smiles}

Chemical Descriptors:
- Molecular Weight: {descriptors['molecular_weight']} Da (ideal: 160-480)
- Lipophilicity (logP): {descriptors['logP']} (ideal: 0-5)
- H-bond Donors: {descriptors['hbd']} (ideal: ≤5)
- H-bond Acceptors: {descriptors['hba']} (ideal: ≤10)
- Topological Polar Surface Area (TPSA): {descriptors['tpsa']} Ų (ideal: <140)
- Rotatable Bonds: {descriptors['rotatable_bonds']} (ideal: <10)

Efficiency Score: {score}/10

Analysis Requirements:
1. Analyze overall chemical descriptors (molecular weight, lipophilicity, etc.)
2. **Focus on head group structure** (e.g., ester, amide functional groups) and its impact on drug efficiency
3. **Focus on tail structure** (e.g., alkyl chains, fatty chains) and its impact on drug efficiency
4. Explain how head and tail structures affect pharmacokinetic properties

Example Format:
"Due to the ester and amide groups at the head and moderate-length fatty chains (C6) at the tail, with molecular weight of 550 Da, the molecule shows good membrane permeability but moderate metabolic stability, resulting in a score of 6.5/10."

Please respond in the format "Due to... (including head and tail structural features), the molecule... (pharmacokinetic effects), resulting in a score of X/10." (80-120 words). Must include specific structural features and numerical values."""

    messages = [
        {
            "role": "system",
            "content": "You are a professional medicinal chemistry expert specializing in molecular structure and drug property analysis.",
        },
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=300,  # room for head/tail discussion
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("\nLoading spreadsheet...")
    df = pd.read_excel(os.path.join(base_dir, "data", "virtual_library.xlsx"))
    print(f"Rows: {len(df)}")

    if TEST_MODE:
        df = df.head(TEST_SIZE)
        print(f"Test mode: first {TEST_SIZE} rows only")

    results = []
    start_time = time.time()

    print("\nProcessing...")
    for _idx, row in df.iterrows():
        number = row["Number"]
        smiles = row["smiles"]

        descriptors = calculate_molecular_descriptors(smiles)

        if descriptors is None:
            print(f"Skipping id {number} (invalid SMILES)")
            continue

        score = calculate_efficiency_score(descriptors)

        print(f"id {number} (score {score}) — generating rationale...")
        reasoning = generate_reasoning(smiles, descriptors, score)

        results.append(
            {
                "number": int(number),
                "smiles": smiles,
                "efficiency_score": score,
                "molecular_weight": descriptors["molecular_weight"],
                "logP": descriptors["logP"],
                "hbd": descriptors["hbd"],
                "hba": descriptors["hba"],
                "tpsa": descriptors["tpsa"],
                "rotatable_bonds": descriptors["rotatable_bonds"],
                "reasoning": reasoning,
            }
        )

    results.sort(key=lambda x: x["efficiency_score"], reverse=True)

    output_file = os.path.join(base_dir, "data", "qwen_verify_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)
    print(f"Wall time: {elapsed:.2f} s")
    print(f"Per compound: {elapsed/len(results):.2f} s")
    print(f"Processed: {len(results)}")
    print(f"Saved: {output_file}")

    score_counts = Counter(r["efficiency_score"] for r in results)
    print("\nScore histogram:")
    for score in sorted(score_counts.keys(), reverse=True):
        count = score_counts[score]
        percentage = count / len(results) * 100
        print(f"  {score:.1f}: {count} ({percentage:.1f}%)")

    print("\n" + "=" * 70)
    print("Top 3 with rationale preview")
    print("=" * 70)
    for i, drug in enumerate(results[:3], 1):
        print(f"\n{i}. id {drug['number']} — score {drug['efficiency_score']}/10")
        print(f"   SMILES: {drug['smiles'][:60]}...")
        print(f"   MW: {drug['molecular_weight']}, logP: {drug['logP']}, TPSA: {drug['tpsa']}")
        print(f"   Rationale: {drug['reasoning']}")
        print("-" * 70)


if __name__ == "__main__":
    main()
