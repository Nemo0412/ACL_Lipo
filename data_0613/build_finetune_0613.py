#!/usr/bin/env python3
"""
Build SFT train/val JSONL from:
  new-20260602-library training data-expansion lipid.xlsx   (414 rows)

Normalization
-------------
  Efficiency (raw, 1.33 – 4320) → 1-10 integer via log10 percentile rank.
  This is consistent with what TxGemma predict_agent uses.

Output format (one JSON object per line, HuggingFace chat-message format)
--------------------------------------------------------------------------
  {
    "messages": [
      {"role": "user",      "content": "<prompt>"},
      {"role": "assistant", "content": "Efficiency Score: 7"}
    ]
  }

No rationale is included (score-only fine-tuning as requested).

Usage
-----
  python data_0613/build_finetune_0613.py
  python data_0613/build_finetune_0613.py --val_ratio 0.15 --seed 123
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import pandas as pd


# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
TRAIN_XLSX = _REPO / "data_0613" / "new-20260602-library training data-expansion lipid.xlsx"
OUT_DIR    = _REPO / "data_0613" / "finetune"


# ── Efficiency → 1-10 ─────────────────────────────────────────────────────────

def efficiency_to_score(series: pd.Series) -> pd.Series:
    """
    Map raw Efficiency values to integer 1-10 via log10 percentile rank.
    Handles zeros and negatives by clipping to a small positive value.
    """
    log_vals = series.clip(lower=1e-6).apply(math.log10)
    pct = log_vals.rank(method="average", pct=True, ascending=True)
    scores = (1 + 9 * pct).clip(1, 10).round().astype(int)
    return scores


# ── Prompt builder ────────────────────────────────────────────────────────────

def make_prompt(row: pd.Series) -> str:
    head = str(row.get("Head", "N/A")).strip()
    tail = str(row.get("Tail", "N/A")).strip()
    smiles = str(row.get("Smiles", row.get("SMILES", ""))).strip()
    return (
        "You are an expert in lipid nanoparticles and mRNA delivery. "
        "Predict the mRNA transfection efficiency score (1-10) for this lipid molecule. "
        "Reply with ONLY 'Efficiency Score: <number>' and nothing else.\n\n"
        f"SMILES: {smiles}\n"
        f"Head group: {head}\n"
        f"Tail group: {tail}"
    )


def make_completion(score: int) -> str:
    return f"Efficiency Score: {score}"


def row_to_example(row: pd.Series) -> dict:
    return {
        "messages": [
            {"role": "user",      "content": make_prompt(row)},
            {"role": "assistant", "content": make_completion(int(row["score_1_10"]))},
        ]
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(TRAIN_XLSX))
    p.add_argument("--out_dir", default=str(OUT_DIR))
    p.add_argument("--val_ratio", type=float, default=0.15,
                   help="Fraction of data to use as validation set")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_excel(args.input)
    print(f"Loaded {len(df)} rows from {os.path.basename(args.input)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Efficiency range: {df['Efficiency'].min():.2f} – {df['Efficiency'].max():.2f}")

    df["score_1_10"] = efficiency_to_score(df["Efficiency"])

    # Distribution check
    print("\nScore distribution (1-10):")
    dist = df["score_1_10"].value_counts().sort_index()
    for s, c in dist.items():
        bar = "█" * int(c / dist.max() * 30)
        print(f"  {s:2d}: {c:3d}  {bar}")

    # Train / val split
    indices = list(df.index)
    random.seed(args.seed)
    random.shuffle(indices)
    n_val = max(1, int(len(indices) * args.val_ratio))
    val_idx   = indices[:n_val]
    train_idx = indices[n_val:]

    def write_jsonl(path: str, rows: pd.DataFrame) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for _, row in rows.iterrows():
                f.write(json.dumps(row_to_example(row), ensure_ascii=False) + "\n")
        print(f"  Wrote {len(rows)} examples → {path}")

    train_path = os.path.join(args.out_dir, "train_0613.jsonl")
    val_path   = os.path.join(args.out_dir, "val_0613.jsonl")

    write_jsonl(train_path, df.loc[train_idx])
    write_jsonl(val_path,   df.loc[val_idx])

    # Print a few examples
    print("\nSample training examples:")
    for _, row in df.loc[train_idx[:2]].iterrows():
        ex = row_to_example(row)
        print(f"  score={row['score_1_10']}  Efficiency={row['Efficiency']:.1f}")
        print(f"  user   : {ex['messages'][0]['content'][:120]}...")
        print(f"  model  : {ex['messages'][1]['content']}")
        print()


if __name__ == "__main__":
    main()
