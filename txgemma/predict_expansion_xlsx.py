#!/usr/bin/env python3
"""
TxGemma-27B-Chat: score mRNA transfection efficiency (1–10) for each lipid in
`Smiles-Expension lipid.xlsx` (or any workbook with SMILES in the last column).

Outputs JSON sorted by efficiency_score (high → low), with periodic checkpoints
for long runs (~10k+ compounds).

Example (cluster):
  sbatch txgemma/run_predict_expansion.sbatch

Example (interactive):
  python txgemma/predict_expansion_xlsx.py \\
      --input /path/to/Smiles-Expension\\ lipid.xlsx \\
      --output data/smiles_expansion_txgemma_scores.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime

# Allow `python txgemma/predict_expansion_xlsx.py` from repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
import torch
import torch._dynamo
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Long generation loops can hit PyTorch Dynamo default graph cache limits (error:
# "cache_size_limit reached"), yielding bogus fallback scores unless raised.
# Generation can compile many cache variants; defaults (8 / 256) trigger
# "cache_size_limit reached" and fail every row. Use generously high caps.
torch._dynamo.config.cache_size_limit = 16384
torch._dynamo.config.accumulated_cache_size_limit = 16384

# Reuse prompt + parsing from the main virtual-library agent
from txgemma.predict_agent import (
    create_prediction_prompt,
    extract_score_and_reason,
)


def _parse_tail_count(val) -> int:
    s = str(val).strip()
    m = re.match(r"(\d+)", s)
    return int(m.group(1)) if m else 1


def load_expansion_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    cols = list(df.columns)
    if not cols:
        raise ValueError("Empty spreadsheet")
    # SMILES live in the last column for this workbook
    smiles_col = cols[-1]
    df = df.rename(columns={smiles_col: "SMILES"})
    return df


def row_to_mol_data(row: pd.Series) -> dict:
    linker = row.get("Unnamed: 4")
    if linker is None or (isinstance(linker, float) and pd.isna(linker)):
        linker_len = 0
    else:
        try:
            linker_len = int(float(linker))
        except (TypeError, ValueError):
            linker_len = 0

    return {
        "ID": int(row["number"]),
        "SMILES": str(row["SMILES"]).strip(),
        "amino_acid": str(row["Head"]),
        "protection_group": str(row["Tail"]),
        "linker_length": linker_len,
        "tail_count": _parse_tail_count(row["tail number"]),
        "ester_bonds_OCOO": 0,
    }


def load_checkpoint(path: str) -> tuple[list[dict], set[int]]:
    if not os.path.exists(path):
        return [], set()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    done = {int(r["ID"]) for r in data}
    return data, done


def save_results(path: str, results: list[dict]) -> None:
    # Always persist sorted by score (descending) for the deliverable
    sorted_results = sorted(results, key=lambda x: x["efficiency_score"], reverse=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sorted_results, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default=None,
        help="Path to Smiles-Expension lipid.xlsx",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output JSON (sorted high→low by efficiency_score)",
    )
    p.add_argument(
        "--checkpoint_every",
        type=int,
        default=50,
        help="Save checkpoint every N compounds",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only first N rows (debug)",
    )
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_path = args.input or os.path.join(base_dir, "Smiles-Expension lipid.xlsx")
    out_path = args.output or os.path.join(
        base_dir, "data", "smiles_expansion_tx27b_scores.json"
    )

    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = load_expansion_dataframe(in_path)
    if args.limit > 0:
        df = df.head(args.limit)

    results, done_ids = load_checkpoint(out_path)
    pending_rows = [row for _, row in df.iterrows() if int(row["number"]) not in done_ids]

    print(f"Input: {in_path}")
    print(f"Rows in sheet: {len(df)}")
    print(f"Already done (resume): {len(done_ids)}")
    print(f"To run: {len(pending_rows)}")
    print(f"Output: {out_path}")

    if not pending_rows:
        print("Nothing to do; all IDs present in output.")
        save_results(out_path, results)
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("Warning: CUDA not available; 27B inference on CPU is not practical.")

    local_model_path = os.path.join(base_dir, "Txgemma", "txgemma-27b-chat")
    model_names = [
        local_model_path,
        "google/txgemma-27b-chat",
        "google/gemma-2-27b-it",
    ]

    tokenizer = None
    model = None
    model_name_used = None

    for model_name in model_names:
        try:
            print(f"Loading: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                token=True,
            )
            model_name_used = model_name
            print("Model ready.")
            break
        except Exception as e:
            print(f"  Failed: {e}")

    if model is None:
        raise RuntimeError("Could not load any TxGemma/Gemma checkpoint.")

    model.eval()
    # With device_map="auto", place tensors on the module's primary device
    infer_device = next(model.parameters()).device

    start = datetime.now()
    since_save = 0

    for row in tqdm(pending_rows, desc="TxGemma expansion"):
        mol = row_to_mol_data(row)
        mol_id = mol["ID"]
        try:
            prompt = create_prediction_prompt(mol)
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            ).to(infer_device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            score, reason = extract_score_and_reason(response)
            results.append(
                {
                    "ID": mol_id,
                    "SMILES": mol["SMILES"],
                    "Head": str(row["Head"]),
                    "Tail": str(row["Tail"]),
                    "tail_number_field": str(row["tail number"]),
                    "efficiency_score": score,
                    "reason": reason,
                }
            )
        except Exception as e:
            results.append(
                {
                    "ID": mol_id,
                    "SMILES": mol["SMILES"],
                    "Head": str(row["Head"]),
                    "Tail": str(row["Tail"]),
                    "tail_number_field": str(row["tail number"]),
                    "efficiency_score": 5,
                    "reason": f"ERROR: {e}",
                }
            )

        since_save += 1
        if since_save >= args.checkpoint_every:
            save_results(out_path, results)
            since_save = 0

    save_results(out_path, results)

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\nFinished in {elapsed/3600:.2f} h ({elapsed/len(pending_rows):.1f} s / compound)")
    print(f"Saved (sorted): {out_path}")

    # Optional CSV for spreadsheets
    csv_path = out_path.replace(".json", ".csv")
    out_df = pd.DataFrame(sorted(results, key=lambda x: x["efficiency_score"], reverse=True))
    out_df.to_csv(csv_path, index=False)
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
