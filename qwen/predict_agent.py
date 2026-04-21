"""
Local Qwen model: predict lipid transfection efficiency scores for the virtual library.
No external API; loads open weights and runs inference.
"""
import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import time
import os

# Model id (override with env QWEN_MODEL if needed)
MODEL_NAME = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-8B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Debug: only first N rows
TEST_MODE = False
TEST_SIZE = 5

# Generation hyperparameters
TEMPERATURE = 0.3  # lower for stabler scores
MAX_NEW_TOKENS = 50
TOP_P = 0.9

print("=" * 60)
print("Efficiency scoring (local Qwen)")
print("=" * 60)
print(f"Model: {MODEL_NAME}")
print(f"Device: {DEVICE}")
print(f"Test mode: {'on' if TEST_MODE else 'off'}")
if TEST_MODE:
    print(f"Test size: first {TEST_SIZE} rows")
print("=" * 60)

# Load tokenizer + weights
print("\nLoading tokenizer and model...")
print("(First run may download weights from Hugging Face.)")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )

    print("Model loaded.")

except Exception as e:
    print(f"\nModel load failed: {e}")
    print("\nTips:")
    print("1. Out of memory: set QWEN_MODEL to a smaller checkpoint, e.g. Qwen/Qwen3-8B")
    print("2. Network: check HF_ENDPOINT / proxy settings")
    print("3. Gated models: run `huggingface-cli login`")
    exit(1)


def create_efficiency_prompt(smiles: str, drug_info: dict) -> str:
    """Build the user prompt for a single compound."""
    return f"""You are a medicinal chemist. Predict the transfection efficiency score (Efficiency Score).

Compound metadata:
- Number: {drug_info.get('Number', 'N/A')}
- Amino acid: {drug_info.get('Amino acid', 'N/A')}
- Protection: {drug_info.get('Protection', 'N/A')}
- Linker: {drug_info.get('linker', 'N/A')}
- OCOO: {drug_info.get('OCOO', 'N/A')}
- Tail: {drug_info.get('tail', 'N/A')}
- SMILES: {smiles}

Scale (integer 0-10):
- 0-2: very poor activity
- 3-4: low activity
- 5-6: moderate activity
- 7-8: good activity
- 9-10: excellent activity

Consider drug-likeness, bioavailability, stability, target engagement, and toxicity risk.

Reply with a single integer from 0 to 10 only, e.g. 7"""


def predict_efficiency(smiles: str, drug_info: dict) -> float:
    """Run one forward pass and parse an integer score."""
    prompt = create_efficiency_prompt(smiles, drug_info)

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a medicinal chemist. Output only one integer from 0 to 10. "
                    "No explanation or chain-of-thought."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"  Model reply: {response.strip()}")

        import re

        response_clean = response.strip()

        try:
            score = int(float(response_clean))
            score = max(0, min(10, score))
            return float(score)
        except ValueError:
            numbers = re.findall(r"\d+", response_clean)
            if numbers:
                score = int(numbers[0])
                score = max(0, min(10, score))
                return float(score)
            print("  Warning: could not parse a number; defaulting to 5")
            return 5.0

    except Exception as e:
        print(f"  Warning: prediction failed: {e}")
        return 5.0


def main():
    """Read Excel, score each row, sort, and write JSON."""

    print("\nReading spreadsheet...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_excel(os.path.join(base_dir, "data", "virtual_library.xlsx"))
    print(f"Rows: {len(df)}")

    if TEST_MODE:
        df = df.head(TEST_SIZE)
        print(f"Test mode: using first {TEST_SIZE} rows only")

    results = []

    print("\nScoring...")
    print("-" * 60)

    start_time = time.time()

    for idx, row in df.iterrows():
        print(f"\n[{idx + 1}/{len(df)}] Compound id: {row['Number']}")

        drug_info = {
            "Number": row["Number"],
            "Amino acid": row["Amino acid"],
            "Protection": row["Protection"],
            "linker": row["linker"],
            "OCOO": row["OCOO"],
            "tail": row["tail"],
        }

        smiles = row["smiles"]
        print(f"  SMILES: {smiles[:60]}..." if len(smiles) > 60 else f"  SMILES: {smiles}")

        efficiency_score = predict_efficiency(smiles, drug_info)
        print(f"  Score: {efficiency_score}")

        result_entry = {
            "number": int(row["Number"]),
            "amino_acid": str(row["Amino acid"]),
            "protection": str(row["Protection"]),
            "linker": str(row["linker"]),
            "OCOO": str(row["OCOO"]),
            "tail": str(row["tail"]),
            "smiles": smiles,
            "efficiency_score": int(efficiency_score),
        }
        results.append(result_entry)

    elapsed_time = time.time() - start_time
    print("\n" + "-" * 60)

    print("\nSorting by score (high to low)...")
    results_sorted = sorted(results, key=lambda x: x["efficiency_score"], reverse=True)

    output_file = os.path.join(base_dir, "data", "qwen_predict_results.json")
    print(f"Writing {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_sorted, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
    print("\nSummary:")
    print(f"- Compounds: {len(results_sorted)}")
    print(f"- Wall time: {elapsed_time:.2f} s")
    print(f"- Per compound: {elapsed_time/len(results_sorted):.2f} s")
    print(
        f"- Mean score: {sum(r['efficiency_score'] for r in results_sorted) / len(results_sorted):.2f}"
    )
    print(f"- Max score: {results_sorted[0]['efficiency_score']} (id {results_sorted[0]['number']})")
    print(f"- Min score: {results_sorted[-1]['efficiency_score']} (id {results_sorted[-1]['number']})")

    print(f"\nTop scores (showing up to 10 of {len(results_sorted)}):")
    for i, drug in enumerate(results_sorted[: min(10, len(results_sorted))], 1):
        print(f"  {i}. id {drug['number']}: {drug['efficiency_score']}")

    if len(results_sorted) > 10:
        print("  ...")

    print(f"\nSaved: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
