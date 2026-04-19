#!/usr/bin/env python3
"""
Prepare training data for TxGemma-9B SFT finetuning.

Input format (CSV or JSONL with ground-truth labels):
  CSV  columns: smiles, efficiency_score, [reasoning]
  JSONL format: {"messages": [...]} (already in chat format, pass through)

Output: train.jsonl / val.jsonl  (conversation format for SFTTrainer)

Usage:
  python prepare_data.py --input data/train_labels.csv --output_dir data/finetune/
  python prepare_data.py --input data/train_labels.jsonl --output_dir data/finetune/
"""

import argparse
import json
import os
import random
import re

import pandas as pd


# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert in lipid nanoparticles (LNP) and mRNA delivery. "
    "Given the SMILES structure of a lipid molecule, predict its mRNA "
    "transfection efficiency on a scale from 1 to 10, where 1-2 is very "
    "poor, 3-4 is poor, 5-6 is moderate, 7-8 is good, and 9-10 is excellent. "
    "Provide the score and a brief chemical rationale."
)

USER_TEMPLATE = (
    "Predict the mRNA transfection efficiency for the following lipid "
    "molecule.\n\nSMILES structure: {smiles}\n\n"
    "Please respond in this exact format:\n"
    "Efficiency Score: [1-10]\n"
    "Reason: [brief rationale]"
)

ASSISTANT_TEMPLATE_WITH_REASON = (
    "Efficiency Score: {score}\n"
    "Reason: {reason}"
)

ASSISTANT_TEMPLATE_SCORE_ONLY = "Efficiency Score: {score}"


# ── Conversion helpers ────────────────────────────────────────────────────────

def make_conversation(smiles: str, score: int, reason: str = "") -> dict:
    user_msg = USER_TEMPLATE.format(smiles=smiles.strip())
    if reason:
        assistant_msg = ASSISTANT_TEMPLATE_WITH_REASON.format(
            score=score, reason=reason.strip()
        )
    else:
        assistant_msg = ASSISTANT_TEMPLATE_SCORE_ONLY.format(score=score)

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def load_csv(path: str) -> list[dict]:
    df = pd.read_csv(path)
    required = {"smiles", "efficiency_score"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(
            f"CSV missing columns: {missing}. "
            f"Found: {df.columns.tolist()}"
        )
    df.columns = df.columns.str.lower()
    records = []
    for _, row in df.iterrows():
        smiles = str(row["smiles"]).strip()
        score  = int(float(row["efficiency_score"]))
        score  = max(1, min(10, score))
        reason = str(row.get("reasoning", row.get("reason", ""))).strip()
        if reason.lower() in ("nan", "none", ""):
            reason = ""
        records.append(make_conversation(smiles, score, reason))
    return records


def load_jsonl_chat(path: str) -> list[dict]:
    """Pass-through for data already in conversation format."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Already in {"messages": [...]} format
            if "messages" in item:
                records.append(item)
            else:
                # Try to extract smiles + score from flat format
                smiles = item.get("smiles", item.get("SMILES", ""))
                score  = item.get("efficiency_score", item.get("score", 5))
                reason = item.get("reason", item.get("reasoning", ""))
                if smiles:
                    records.append(make_conversation(smiles, int(score), str(reason)))
    return records


def load_input(path: str) -> list[dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv"):
        return load_csv(path)
    elif ext in (".jsonl", ".json"):
        return load_jsonl_chat(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def split_and_save(records: list[dict], output_dir: str,
                   val_ratio: float = 0.1, seed: int = 42):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    random.shuffle(records)

    n_val   = max(1, int(len(records) * val_ratio))
    val_set = records[:n_val]
    trn_set = records[n_val:]

    for split, data in [("train", trn_set), ("val", val_set)]:
        out_path = os.path.join(output_dir, f"{split}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved {len(data):>5d} examples → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare SFT training data")
    parser.add_argument("--input",      required=True,
                        help="Path to labeled data (CSV or JSONL)")
    parser.add_argument("--output_dir", default="data/finetune",
                        help="Directory to write train.jsonl / val.jsonl")
    parser.add_argument("--val_ratio",  type=float, default=0.1,
                        help="Fraction of data held out for validation")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    print(f"Loading data from: {args.input}")
    records = load_input(args.input)
    print(f"Total examples: {len(records)}")

    split_and_save(records, args.output_dir, args.val_ratio, args.seed)
    print("Done.")


if __name__ == "__main__":
    main()
