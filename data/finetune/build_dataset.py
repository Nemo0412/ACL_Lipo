#!/usr/bin/env python3
"""
Prepare SFT training data from Qwen + TxGemma prediction files.

Combines:
  - efficiency_scores_ranked_local.json  (Qwen Predict Agent labels, 2588 samples)
  - txgemma_labels.json                  (TxGemma Predict Agent labels, 10K samples + reasoning)

Output: data/finetune/train.jsonl  /  data/finetune/val.jsonl
"""

import json
import os
import random
from collections import Counter

SYSTEM_PROMPT = (
    "You are an expert in lipid nanoparticles (LNP) and mRNA delivery. "
    "Given the SMILES structure and structural features of a lipid molecule, "
    "predict its mRNA transfection efficiency on a scale from 1 to 10 "
    "(1-2: very poor, 3-4: poor, 5-6: moderate, 7-8: good, 9-10: excellent). "
    "Provide the integer efficiency score and a brief chemical rationale."
)

USER_TEMPLATE = (
    "Predict the mRNA transfection efficiency for the following lipid molecule.\n\n"
    "SMILES: {smiles}\n"
    "Amino acid: {amino_acid}\n"
    "Protection group: {protection}\n"
    "Linker length: {linker}\n"
    "OCOO ester bonds: {ocoo}\n"
    "Number of tails: {tail}\n\n"
    "Respond in this exact format:\n"
    "Efficiency Score: [1-10]\n"
    "Reason: [brief chemical rationale]"
)

USER_TEMPLATE_SIMPLE = (
    "Predict the mRNA transfection efficiency for the following lipid molecule.\n\n"
    "SMILES: {smiles}\n\n"
    "Respond in this exact format:\n"
    "Efficiency Score: [1-10]\n"
    "Reason: [brief chemical rationale]"
)


def make_conv(smiles, score, reason="", amino_acid="", protection="",
              linker="", ocoo="", tail=""):
    score = max(1, min(10, round(float(score))))
    if amino_acid:
        user_msg = USER_TEMPLATE.format(
            smiles=smiles, amino_acid=amino_acid, protection=protection,
            linker=linker, ocoo=ocoo, tail=tail,
        )
    else:
        user_msg = USER_TEMPLATE_SIMPLE.format(smiles=smiles)

    if reason and reason.strip().lower() not in ("", "nan", "none"):
        assistant_msg = f"Efficiency Score: {score}\nReason: {reason.strip()}"
    else:
        assistant_msg = f"Efficiency Score: {score}"

    return {"messages": [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]}


def load_qwen_data(path):
    """Load Leshu_Exp/data/efficiency_scores_ranked_local.json"""
    with open(path) as f:
        data = json.load(f)
    records = []
    for d in data:
        records.append(make_conv(
            smiles=d["smiles"],
            score=d["efficiency_score"],
            amino_acid=str(d.get("amino_acid", "")),
            protection=str(d.get("protection", "")),
            linker=str(d.get("linker", "")),
            ocoo=str(d.get("OCOO", "")),
            tail=str(d.get("tail", "")),
        ))
    return records


def load_txgemma_data(path):
    """Load Texgemma/result.json  (has SMILES, efficiency_score, reason)"""
    with open(path) as f:
        data = json.load(f)
    records = []
    for d in data:
        records.append(make_conv(
            smiles=d.get("SMILES", d.get("smiles", "")),
            score=d["efficiency_score"],
            reason=d.get("reason", ""),
        ))
    return records


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    out_dir = base  # same folder: data/finetune/

    random.seed(42)

    # ── Load both datasets ─────────────────────────────────────────────────
    qwen_path    = os.path.join(base, "qwen_labels.json")
    txgemma_path = os.path.join(base, "txgemma_labels.json")

    qwen_records    = load_qwen_data(qwen_path)    if os.path.exists(qwen_path)    else []
    txgemma_records = load_txgemma_data(txgemma_path) if os.path.exists(txgemma_path) else []

    print(f"Qwen records    : {len(qwen_records):>6}  (from {qwen_path})")
    print(f"TxGemma records : {len(txgemma_records):>6}  (from {txgemma_path})")

    # ── Score distribution check ───────────────────────────────────────────
    all_records = txgemma_records + qwen_records
    dist = Counter(
        r["messages"][2]["content"].split("Efficiency Score: ")[1].split("\n")[0]
        for r in all_records
    )
    print(f"\nCombined score distribution (total {len(all_records)}):")
    for s in sorted(dist.keys(), key=lambda x: float(x)):
        bar = "█" * (dist[s] // 50)
        print(f"  score {s:>4}: {dist[s]:>5}  {bar}")

    # ── Train / val split ──────────────────────────────────────────────────
    random.shuffle(all_records)
    n_val = max(50, int(len(all_records) * 0.05))
    val_set   = all_records[:n_val]
    train_set = all_records[n_val:]

    for split, data in [("train", train_set), ("val", val_set)]:
        out_path = os.path.join(out_dir, f"{split}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(data):>6} examples → {out_path}")

    print("\nDone. Ready for finetuning.")


if __name__ == "__main__":
    main()
