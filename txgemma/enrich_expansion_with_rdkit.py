#!/usr/bin/env python3
"""
Add RDKit descriptors + rdkit_efficiency_score (1–10) to expansion prediction JSON.

Scores use percentile rank within the file so the full library separates into bands.

Example:
  python txgemma/enrich_expansion_with_rdkit.py \\
    --input data/smiles_expansion_tx27b_scores.json \\
    --output data/smiles_expansion_tx27b_scores_with_rdkit.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd
from tqdm import tqdm

from txgemma.rdkit_lipid_score import (
    compute_row_from_smiles,
    enrich_record_rdkit,
    percentile_rank_scores_1_10,
)


def _tx_failed(rec: dict) -> bool:
    return str(rec.get("reason", "")).startswith("ERROR:")


def _combined_score(rec: dict) -> int | None:
    tx = rec.get("efficiency_score")
    rk = rec.get("rdkit_efficiency_score")
    if rk is None:
        return None
    if _tx_failed(rec) or tx is None:
        return int(rk)
    try:
        c = round(0.5 * int(tx) + 0.5 * int(rk))
        return int(max(1, min(10, c)))
    except (TypeError, ValueError):
        return int(rk)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Existing JSON (e.g. txgemma output)")
    p.add_argument("--output", default=None, help="Output JSON path")
    p.add_argument(
        "--sort-by",
        choices=("rdkit_efficiency_score", "efficiency_score", "combined_efficiency_score"),
        default="rdkit_efficiency_score",
    )
    args = p.parse_args()

    out_path = args.output or args.input.replace(".json", "_with_rdkit.json")

    with open(args.input, encoding="utf-8") as f:
        records: list[dict] = json.load(f)

    raws: list[float | None] = []
    metas: list[tuple[dict | None, str | None]] = []

    for rec in tqdm(records, desc="RDKit descriptors"):
        smi = rec.get("SMILES", "")
        raw, desc, err = compute_row_from_smiles(str(smi))
        raws.append(raw)
        metas.append((desc, err))

    rk_scores = percentile_rank_scores_1_10(raws)

    enriched: list[dict] = []
    for rec, raw, (desc, err), rks in zip(records, raws, metas, rk_scores):
        row = dict(rec)
        row.update(enrich_record_rdkit(raw, desc, rks, err))
        row["combined_efficiency_score"] = _combined_score({**row})
        enriched.append(row)

    sort_key = args.sort_by

    def _key(r: dict) -> tuple:
        v = r.get(sort_key)
        if v is None:
            return (-1,)  # sort last when descending
        return (int(v),)

    enriched.sort(key=_key, reverse=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out_path)

    csv_path = out_path.replace(".json", ".csv")
    pd.DataFrame(enriched).to_csv(csv_path, index=False)

    print(f"Wrote {out_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
