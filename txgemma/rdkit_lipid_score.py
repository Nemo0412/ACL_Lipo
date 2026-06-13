"""
RDKit-derived heuristic scores (1–10) for lipid-like mRNA delivery candidates.

This is a *structure-only proxy* (logP, size, polarity, flexibility, sp3 fraction),
not experimental transfection. Scores are spread across the library using percentile
rank so you get discrimination even when absolute chemistry is similar.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def smiles_to_mol(smiles: str) -> Chem.Mol | None:
    if smiles is None:
        return None
    s = str(smiles).strip()
    if not s or s.lower() == "nan":
        return None
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return m


def compute_lipid_descriptors(mol: Chem.Mol) -> dict[str, float]:
    ha = max(mol.GetNumHeavyAtoms(), 1)
    return {
        "MolWt": float(Descriptors.MolWt(mol)),
        "MolLogP": float(Crippen.MolLogP(mol)),
        "TPSA": float(Descriptors.TPSA(mol)),
        "NumHAcceptors": float(Lipinski.NumHAcceptors(mol)),
        "NumHDonors": float(Lipinski.NumHDonors(mol)),
        "NumRotatableBonds": float(Lipinski.NumRotatableBonds(mol)),
        "HeavyAtomCount": float(ha),
        "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "NumRings": float(rdMolDescriptors.CalcNumRings(mol)),
        "RotPerHeavy": float(Lipinski.NumRotatableBonds(mol)) / float(ha),
    }


def lipid_delivery_raw_score(desc: dict[str, float]) -> float:
    """
    Higher ≈ more lipid nanoparticle–friendly *in silico* (hydrophobic tails,
    moderate polarity, reasonable size, alkyl character).
    """
    logp = desc["MolLogP"]
    mw = desc["MolWt"]
    tpsa = desc["TPSA"]
    rot_per = desc["RotPerHeavy"]
    frac_sp3 = desc["FractionCSP3"]

    lipophile = _clamp01((logp - 4.0) / 16.0)
    size_fit = math.exp(-((mw - 850.0) / 480.0) ** 2)
    # Amphiphilic sweet spot: not bare hydrocarbon, not too polar
    tpsa_term = _clamp01(1.0 - min(abs(tpsa - 110.0) / 150.0, 1.0))
    flex = 1.0 - _clamp01(rot_per * 4.5)
    tail_like = frac_sp3

    return (
        0.28 * lipophile
        + 0.22 * size_fit
        + 0.20 * tpsa_term
        + 0.17 * flex
        + 0.13 * tail_like
    )


def percentile_rank_scores_1_10(values: list[float | None]) -> list[int | None]:
    """
    Map each valid value to 1–10 by percentile rank (higher raw → higher score).
    Invalid (None / nan) entries get None.
    """
    n = len(values)
    out: list[int | None] = [None] * n
    valid_idx = [
        i
        for i, v in enumerate(values)
        if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))
    ]
    if not valid_idx:
        return out
    arr = np.array([float(values[i]) for i in valid_idx], dtype=float)
    pct = pd.Series(arr).rank(method="average", pct=True, ascending=True).to_numpy(dtype=float)
    scored = np.clip(np.ceil(1.0 + 9.0 * pct), 1, 10).astype(int)
    for k, i in enumerate(valid_idx):
        out[i] = int(scored[k])
    return out


def enrich_record_rdkit(
    raw: float | None,
    desc: dict[str, float] | None,
    rdkit_score: int | None,
    err: str | None,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "rdkit_smiles_ok": err is None and desc is not None,
        "rdkit_raw_score": raw,
        "rdkit_efficiency_score": rdkit_score,
        "rdkit_error": err,
    }
    if desc is not None:
        block["rdkit_descriptors"] = {k: round(v, 6) for k, v in desc.items()}
    return block


def compute_row_from_smiles(smiles: str) -> tuple[float | None, dict[str, float] | None, str | None]:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None, None, "RDKit_parse_failed"
    desc = compute_lipid_descriptors(mol)
    return lipid_delivery_raw_score(desc), desc, None
