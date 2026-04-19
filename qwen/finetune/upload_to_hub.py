#!/usr/bin/env python3
"""
Merge LoRA adapter into base TxGemma-9B weights and push to HuggingFace Hub.

Steps:
  1. Load base model + LoRA adapter
  2. Merge LoRA weights into base model
  3. Save merged model locally
  4. Push to HuggingFace Hub

Usage:
  python upload_to_hub.py \
      --base_model   Qwen/Qwen2.5-32B-Instruct \
      --adapter_path checkpoints/qwen-32b-lipo/final_adapter \
      --hf_repo      YOUR_HF_USERNAME/qwen-32b-lipo-finetuned \
      --merged_dir   checkpoints/qwen-32b-lipo/merged
"""

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_CARD = """\
---
license: gemma
base_model: Qwen/Qwen2.5-32B-Instruct
tags:
  - biology
  - chemistry
  - lipid-nanoparticle
  - mRNA-delivery
  - LoRA
  - finetuned
---

# TxGemma-9B-LiPo (Finetuned)

This model is **TxGemma-9B-Chat** finetuned with LoRA on a lipid nanoparticle
(LNP) mRNA transfection efficiency dataset for the ACL paper submission.

## Task

Given the SMILES structure of a lipid molecule, predict its mRNA transfection
efficiency score (1–10) and provide a chemical rationale.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

messages = [
    {{
        "role": "system",
        "content": (
            "You are an expert in lipid nanoparticles (LNP) and mRNA delivery. "
            "Given the SMILES structure of a lipid molecule, predict its mRNA "
            "transfection efficiency on a scale from 1 to 10 and provide a brief rationale."
        ),
    }},
    {{
        "role": "user",
        "content": "Predict the mRNA transfection efficiency for the following lipid molecule.\\n\\n"
                   "SMILES structure: CC(=O)OCC\\n\\n"
                   "Please respond in this exact format:\\n"
                   "Efficiency Score: [1-10]\\n"
                   "Reason: [brief rationale]",
    }},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

## Training Details

- **Base model**: Qwen/Qwen2.5-32B-Instruct
- **Method**: LoRA (r=16, α=32) on all attention + FFN projection layers
- **Dataset**: LNP virtual library with ground-truth mRNA transfection labels
- **Hardware**: NVIDIA H200 GPU
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",   default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--adapter_path", required=True,
                   help="Path to saved LoRA adapter (final_adapter/)")
    p.add_argument("--hf_repo",      required=True,
                   help="HuggingFace repo ID, e.g. YOUR_USER/qwen-32b-lipo")
    p.add_argument("--merged_dir",   default=None,
                   help="Local dir to save merged weights (optional)")
    p.add_argument("--private",      action="store_true",
                   help="Make the HF repo private")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*70}")
    print("TxGemma-9B: Merge LoRA + Upload to HuggingFace Hub")
    print(f"{'='*70}")
    print(f"  Base model  : {args.base_model}")
    print(f"  Adapter     : {args.adapter_path}")
    print(f"  HF repo     : {args.hf_repo}")

    # ── 1. Load base model ────────────────────────────────────────────────────
    print("\n[1/4] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # ── 2. Load LoRA adapter and merge ────────────────────────────────────────
    print("[2/4] Merging LoRA adapter into base weights...")
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    merged_model = peft_model.merge_and_unload()
    print("  Merge complete.")

    # ── 3. Save merged model locally (optional) ───────────────────────────────
    if args.merged_dir:
        print(f"[3/4] Saving merged model to: {args.merged_dir}")
        os.makedirs(args.merged_dir, exist_ok=True)
        merged_model.save_pretrained(args.merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.merged_dir)

        # Write model card
        model_card_path = os.path.join(args.merged_dir, "README.md")
        with open(model_card_path, "w") as f:
            f.write(MODEL_CARD.format(repo_id=args.hf_repo))
        print(f"  Model card written: {model_card_path}")
    else:
        print("[3/4] Skipping local save (--merged_dir not set).")

    # ── 4. Push to HuggingFace Hub ────────────────────────────────────────────
    print(f"[4/4] Pushing to HuggingFace Hub: {args.hf_repo}")
    print("  (This may take several minutes for 9B weights ~18 GB)")

    merged_model.push_to_hub(
        args.hf_repo,
        private=args.private,
        safe_serialization=True,
        commit_message="Upload TxGemma-9B finetuned on LNP efficiency data",
    )
    tokenizer.push_to_hub(
        args.hf_repo,
        private=args.private,
        commit_message="Upload tokenizer",
    )

    # Write model card to hub
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=MODEL_CARD.format(repo_id=args.hf_repo).encode(),
        path_in_repo="README.md",
        repo_id=args.hf_repo,
        repo_type="model",
        commit_message="Add model card",
    )

    print(f"\n✅ Successfully uploaded to: https://huggingface.co/{args.hf_repo}")


if __name__ == "__main__":
    main()
