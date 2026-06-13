#!/usr/bin/env python3
"""
Embedding-space two-pass agent for mRNA transfection efficiency scoring.

Instead of decode → text → re-encode between passes, Pass 1 returns the
model's last-layer hidden states directly and Pass 2 prepends them as
`inputs_embeds` (continuous context) before the verification prompt.

This removes the tokenisation bottleneck: no information is lost
converting floating-point hidden states to discrete tokens.

Architecture
------------
Pass 1  (Predict)
    input_ids  →  TxGemma forward  →  hidden_states[-1]   [1, S, H]
                                   →  logits              used for quick score
Pass 2  (Verify, only for high/low scorers)
    [Pass-1 hidden | Verify prompt embeds]  →  TxGemma generate  →  decode

Both passes share the same loaded model weights.

Usage (interactive)
-------------------
    python txgemma/embedding_agent.py \\
        --input  data/virtual_library_preprocessed.jsonl \\
        --output data/txgemma_embedding_results.json

Usage (SLURM)
-------------
    sbatch txgemma/run_embedding_agent.sbatch
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch._dynamo
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Raise Dynamo graph-cache caps to avoid "cache_size_limit reached" during
# repeated generate() calls over thousands of molecules.
torch._dynamo.config.cache_size_limit = 16384
torch._dynamo.config.accumulated_cache_size_limit = 16384

from txgemma.predict_agent import create_prediction_prompt, extract_score_and_reason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_checkpoint(path: str) -> tuple[list[dict], set]:
    if not os.path.exists(path):
        return [], set()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data, {r["ID"] for r in data}


def _save(path: str, results: list[dict]) -> None:
    sorted_r = sorted(results, key=lambda x: x["efficiency_score"], reverse=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sorted_r, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _digit_token_ids(tokenizer) -> list[int]:
    """Return the first token-ID for each digit string '1'..'10'."""
    ids = []
    for d in range(1, 11):
        enc = tokenizer.encode(str(d), add_special_tokens=False)
        ids.append(enc[0] if enc else 0)
    return ids


# ---------------------------------------------------------------------------
# Pass 1 — Predict (embedding output)
# ---------------------------------------------------------------------------

def predict_pass(
    model,
    tokenizer,
    mol_data: dict,
    device: torch.device,
    digit_ids: list[int],
) -> tuple[int, torch.Tensor]:
    """
    Run a single forward pass and return:
      - quick_score (int 1-10): read from last-token logits over digit tokens
      - hidden      (Tensor [1, seq_len, hidden_dim]): last-layer hidden states

    No text is decoded; information travels as floating-point tensors.
    """
    prompt = create_prediction_prompt(mol_data, need_detailed_reason=False) \
        if _accepts_detailed_kwarg(create_prediction_prompt) \
        else create_prediction_prompt(mol_data)

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(device)

    with torch.no_grad():
        out = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    # --- quick score from last-token logits (no decode needed) ---
    last_logits = out.logits[:, -1, :]               # [1, vocab]
    digit_tensor = torch.tensor(digit_ids, device=device)
    digit_logits = last_logits[0, digit_tensor]       # [10]
    quick_score = int(digit_logits.argmax().item()) + 1  # 1 → 10

    # --- hidden states of the full input sequence ---
    hidden = out.hidden_states[-1]                    # [1, seq_len, H]

    return quick_score, hidden


def _accepts_detailed_kwarg(fn) -> bool:
    import inspect
    return "need_detailed_reason" in inspect.signature(fn).parameters


# ---------------------------------------------------------------------------
# Pass 2 — Verify (embedding input)
# ---------------------------------------------------------------------------

def verify_pass(
    model,
    tokenizer,
    predict_hidden: torch.Tensor,
    mol_data: dict,
    device: torch.device,
) -> tuple[int, str]:
    """
    Prepend Pass-1 hidden states as prefix embeddings before the verify prompt.

    Model sees:  [Pass-1 continuous context | Verify prompt tokens]
    This skips the decode → re-encode round-trip entirely.
    """
    # Build verify prompt and convert to embeddings
    verify_prompt = create_prediction_prompt(mol_data, need_detailed_reason=True) \
        if _accepts_detailed_kwarg(create_prediction_prompt) \
        else create_prediction_prompt(mol_data)

    verify_ids = tokenizer(
        verify_prompt, return_tensors="pt", truncation=True, max_length=1024
    ).input_ids.to(device)

    embed_layer = model.get_input_embeddings()
    verify_embeds = embed_layer(verify_ids)           # [1, verify_len, H]

    # Concatenate: [Pass-1 hidden | verify prompt embeds]
    combined = torch.cat([predict_hidden, verify_embeds], dim=1)  # [1, total, H]
    attn_mask = torch.ones(combined.shape[:2], dtype=torch.long, device=device)

    with torch.no_grad():
        gen_ids = model.generate(
            inputs_embeds=combined,
            attention_mask=attn_mask,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return extract_score_and_reason(response)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(
    model,
    tokenizer,
    molecules: list[dict],
    device: torch.device,
    output_path: str,
    checkpoint_every: int = 50,
    verify_threshold: int = 7,
) -> list[dict]:
    """
    For each molecule:
      Pass 1  →  quick_score + hidden states  (no text decoding)
      Pass 2  →  only if score >= verify_threshold or score == 1
                 hidden states used directly as context (no re-encoding)
    """
    results, done_ids = _load_checkpoint(output_path)
    if done_ids:
        print(f"Resume: {len(done_ids)} already done, skipping.")
    pending = [m for m in molecules if m["ID"] not in done_ids]
    print(f"To process: {len(pending)}")

    digit_ids = _digit_token_ids(tokenizer)
    model.eval()
    infer_device = next(model.parameters()).device
    since_save = 0
    detailed_count = 0

    for mol in tqdm(pending, desc="EmbeddingAgent"):
        mol_id = mol["ID"]
        try:
            # ── Pass 1: predict via forward (embedding output) ──────────────
            quick_score, hidden = predict_pass(
                model, tokenizer, mol, infer_device, digit_ids
            )

            # ── Pass 2: verify via generate (embedding input) ───────────────
            if quick_score >= verify_threshold or quick_score == 1:
                detailed_count += 1
                final_score, reason = verify_pass(
                    model, tokenizer, hidden, mol, infer_device
                )
            else:
                final_score, reason = quick_score, ""

            results.append(
                {
                    "ID": mol_id,
                    "SMILES": mol.get("SMILES", ""),
                    "efficiency_score": final_score,
                    "quick_score_p1": quick_score,
                    "reason": reason,
                    "pass2_used": quick_score >= verify_threshold or quick_score == 1,
                }
            )

        except Exception as exc:
            results.append(
                {
                    "ID": mol_id,
                    "SMILES": mol.get("SMILES", ""),
                    "efficiency_score": 5,
                    "quick_score_p1": None,
                    "reason": f"ERROR: {exc}",
                    "pass2_used": False,
                }
            )

        since_save += 1
        if since_save >= checkpoint_every:
            _save(output_path, results)
            since_save = 0

    _save(output_path, results)
    p2_pct = 100 * detailed_count / max(len(pending), 1)
    print(f"Pass-2 (embed verify) used for {detailed_count}/{len(pending)} molecules ({p2_pct:.1f}%)")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Embedding-space two-pass TxGemma agent")
    p.add_argument("--input", default=None, help="JSONL molecules file")
    p.add_argument("--output", default=None, help="Output JSON (sorted high→low)")
    p.add_argument("--checkpoint_every", type=int, default=50)
    p.add_argument("--verify_threshold", type=int, default=7,
                   help="Pass-2 triggered when quick_score >= this value or == 1")
    p.add_argument("--limit", type=int, default=0, help="Debug: first N molecules only")
    args = p.parse_args()

    base = Path(_REPO_ROOT)
    in_path  = args.input  or str(base / "data" / "virtual_library_preprocessed.jsonl")
    out_path = args.output or str(base / "data" / "txgemma_embedding_results.json")

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input not found: {in_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    molecules = _load_jsonl(in_path)
    if args.limit > 0:
        molecules = molecules[: args.limit]
    print(f"Loaded {len(molecules)} molecules from {in_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    local_path = str(base / "Txgemma" / "txgemma-27b-chat")
    candidates = [local_path, "google/txgemma-27b-chat", "google/gemma-2-27b-it"]

    model = tokenizer = None
    for name in candidates:
        try:
            print(f"Loading: {name}")
            tokenizer = AutoTokenizer.from_pretrained(name, token=True)
            model = AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                token=True,
            )
            print("Model ready.")
            break
        except Exception as exc:
            print(f"  Failed: {exc}")

    if model is None:
        raise RuntimeError("Could not load any TxGemma/Gemma checkpoint.")

    start = datetime.now()
    results = run(
        model, tokenizer, molecules, device, out_path,
        checkpoint_every=args.checkpoint_every,
        verify_threshold=args.verify_threshold,
    )
    elapsed = (datetime.now() - start).total_seconds()
    n = len(results)
    print(f"\nDone in {elapsed/3600:.2f} h ({elapsed/max(n,1):.1f} s/mol)")
    print(f"Saved: {out_path}")

    # Score distribution
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
