#!/usr/bin/env python3
"""
Score all lipids in 'Smiles for lipid screening-19000.xlsx' using a
fine-tuned (LoRA) TxGemma model.

Outputs JSON + CSV sorted by efficiency_score (high → low).
No rationale is generated (score-only, fast inference).

Usage
-----
  python data_0613/score_19k.py \\
      --adapter checkpoints/txgemma-27b-lipo-0613/final_adapter \\
      --output  data_0613/scored_19k.json

Resume support: if --output already exists, rows with existing IDs are skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pandas as pd
import torch
import torch._dynamo
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

torch._dynamo.config.cache_size_limit = 16384
torch._dynamo.config.accumulated_cache_size_limit = 16384

# Suppress RDKit parse warnings (SMILES that score 0 silently)
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass


SCREEN_XLSX = _REPO / "data_0613" / "Smiles for lipid screening-19000.xlsx"
DEFAULT_OUT  = _REPO / "data_0613" / "scored_19k.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_checkpoint(path: str) -> tuple[list[dict], set]:
    if not os.path.exists(path):
        return [], set()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data, {int(r["ID"]) for r in data}


def _save(path: str, results: list[dict]) -> None:
    sorted_r = sorted(results, key=lambda x: x["efficiency_score"], reverse=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sorted_r, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    csv_path = path.replace(".json", ".csv")
    pd.DataFrame(sorted_r).to_csv(csv_path, index=False)


def make_prompt(row: pd.Series) -> str:
    head   = str(row.get("Head", "N/A")).strip()
    tail   = str(row.get("Tail", "N/A")).strip()
    smiles = str(row.get("SMILES", "")).strip()
    return (
        "You are an expert in lipid nanoparticles and mRNA delivery. "
        "Predict the mRNA transfection efficiency score (1-10) for this lipid molecule. "
        "Reply with ONLY 'Efficiency Score: <number>' and nothing else.\n\n"
        f"SMILES: {smiles}\n"
        f"Head group: {head}\n"
        f"Tail group: {tail}"
    )


def extract_score(text: str) -> int:
    for pat in [
        r'[Ee]fficiency\s+[Ss]core\s*[:：]\s*(\d+)',
        r'[Ss]core\s*[:：]\s*(\d+)',
        r'\b([1-9]|10)\b',
    ]:
        m = re.search(pat, text)
        if m:
            s = int(m.group(1))
            if 1 <= s <= 10:
                return s
    return 5  # fallback


# ── Inference ─────────────────────────────────────────────────────────────────

def score_batch(
    model,
    tokenizer,
    df: pd.DataFrame,
    done_ids: set,
    output_path: str,
    checkpoint_every: int = 100,
) -> list[dict]:
    results, _ = _load_checkpoint(output_path)
    infer_device = next(model.parameters()).device
    model.eval()
    since_save = 0

    pending = df[~df["Number"].astype(int).isin(done_ids)]
    print(f"  Total rows: {len(df)}, already done: {len(done_ids)}, pending: {len(pending)}")

    for _, row in tqdm(pending.iterrows(), total=len(pending), desc="Scoring 19k"):
        mol_id = int(row["Number"])
        smiles = str(row.get("SMILES", "")).strip()
        try:
            prompt = make_prompt(row)
            msgs = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=1024
            ).to(infer_device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,   # score only — very short
                    do_sample=False,     # greedy for reproducibility
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            score = extract_score(response)
        except Exception as exc:
            score = 5
            response = f"ERROR: {exc}"

        results.append(
            {
                "ID": mol_id,
                "SMILES": smiles,
                "Head": str(row.get("Head", "")),
                "Tail": str(row.get("Tail", "")),
                "tail_number": str(row.get("tail number", "")),
                "efficiency_score": score,
            }
        )
        since_save += 1
        if since_save >= checkpoint_every:
            _save(output_path, results)
            since_save = 0

    _save(output_path, results)
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter",
                   default=str(_REPO / "checkpoints" / "txgemma-27b-lipo-0613" / "final_adapter"),
                   help="Path to LoRA adapter (fine-tuned). If missing, falls back to base model.")
    p.add_argument("--base_model", default="google/txgemma-27b-chat",
                   help="Base model HF ID or local path")
    p.add_argument("--input",  default=str(SCREEN_XLSX))
    p.add_argument("--output", default=str(DEFAULT_OUT))
    p.add_argument("--checkpoint_every", type=int, default=100)
    p.add_argument("--limit", type=int, default=0, help="Debug: first N rows only")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Load screening data
    df = pd.read_excel(args.input, sheet_name="Sheet1")
    df = df.rename(columns={df.columns[-1]: "SMILES"})   # last col is SMILES
    if args.limit > 0:
        df = df.head(args.limit)
    print(f"Screening library: {len(df)} lipids from {os.path.basename(args.input)}")

    _, done_ids = _load_checkpoint(args.output)

    # Load model
    use_adapter = os.path.isdir(args.adapter)
    base = args.base_model
    local_base = str(_REPO / "Txgemma" / "txgemma-27b-chat")
    if os.path.isdir(local_base):
        base = local_base

    print(f"\nLoading base model: {base}")
    tokenizer = AutoTokenizer.from_pretrained(base, token=True)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=True,
    )

    if use_adapter:
        from peft import PeftModel
        print(f"Loading LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()   # fold LoRA weights for faster inference
        print("Adapter merged.")
    else:
        print(f"WARNING: Adapter not found at {args.adapter}. Using base TxGemma (untuned).")

    # Run scoring
    start = datetime.now()
    results = score_batch(
        model, tokenizer, df, done_ids, args.output,
        checkpoint_every=args.checkpoint_every,
    )
    elapsed = (datetime.now() - start).total_seconds()
    n = len(results)
    print(f"\nFinished in {elapsed/3600:.2f} h ({elapsed/max(n,1):.1f} s/mol)")
    print(f"Output: {args.output}")
    print(f"CSV:    {args.output.replace('.json', '.csv')}")

    # Distribution
    dist: dict[int, int] = {}
    for r in results:
        s = r["efficiency_score"]
        dist[s] = dist.get(s, 0) + 1
    print("\nScore distribution:")
    for s in sorted(dist, reverse=True):
        bar = "█" * int(dist[s] / max(dist.values()) * 30)
        print(f"  {s:2d}: {dist[s]:5d}  {bar}")


if __name__ == "__main__":
    main()
