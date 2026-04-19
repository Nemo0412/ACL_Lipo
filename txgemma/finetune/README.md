# TxGemma-9B Finetuning Pipeline

Full pipeline: prepare data → LoRA SFT → merge weights → upload to HuggingFace.

## Prerequisites

**1. HuggingFace token** with access to `google/txgemma-9b-chat` (gated model).
Get it at https://huggingface.co/settings/tokens then accept the model's license
at https://huggingface.co/google/txgemma-9b-chat

**2. Training data** — a CSV or JSONL file with ground-truth efficiency labels.

CSV format (minimum required columns):
```
smiles,efficiency_score
CC(=O)OCC...,7
...
```
Optional extra column: `reasoning` (free-text rationale).

JSONL format (already in conversation format, pass-through):
```json
{"messages": [{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"Efficiency Score: 7\nReason: ..."}]}
```

---

## Step-by-step

### Step 1 — Prepare training data

```bash
python txgemma/finetune/prepare_data.py \
    --input      data/train_labels.csv \
    --output_dir data/finetune/ \
    --val_ratio  0.1
```
Produces `data/finetune/train.jsonl` and `data/finetune/val.jsonl`.

### Step 2 — Edit the SLURM script

Open `txgemma/finetune/run_finetune.sbatch` and fill in:
```bash
HF_TOKEN="hf_your_token_here"
HF_REPO="your_username/txgemma-9b-lipo-finetuned"
```
Adjust `--account` and `--partition` if needed.

### Step 3 — Submit the SLURM job

```bash
sbatch txgemma/finetune/run_finetune.sbatch
```

This single job will:
1. Authenticate with HuggingFace
2. Download `google/txgemma-9b-chat` to `/scratch/ll5914/models/`
3. Run LoRA finetuning (~2–4 hours on H200)
4. Merge LoRA weights into base model
5. Upload merged model to your HF repo

Monitor with:
```bash
squeue --me
tail -f /scratch/ll5914/logs/txgemma_finetune_<JOBID>.out
```

---

## Files

| File | Description |
|------|-------------|
| `prepare_data.py` | Convert CSV/JSONL labels → SFT conversation format |
| `finetune_sft.py` | LoRA finetuning with HuggingFace Trainer |
| `upload_to_hub.py` | Merge LoRA adapter → push to HuggingFace Hub |
| `run_finetune.sbatch` | SLURM job that runs all steps end-to-end |

## LoRA Configuration (defaults)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `lora_r` | 16 | LoRA rank |
| `lora_alpha` | 32 | LoRA scaling |
| `lora_dropout` | 0.05 | |
| `target_modules` | q,k,v,o,gate,up,down | All attention + FFN |
| `epochs` | 3 | |
| `learning_rate` | 2e-4 | |
| `batch_size` | 2 × 4 accum = eff. 8 | per GPU |
| `max_length` | 1024 tokens | |
| `dtype` | bfloat16 | |
| `GPU` | H200 141GB | ~25 GB used |

## Manual run (without SLURM)

```bash
# Download model
python txgemma/finetune/download_9b.py   # uses HF_TOKEN env var

# Finetune
python txgemma/finetune/finetune_sft.py \
    --model_path /scratch/ll5914/models/txgemma-9b-chat \
    --train_file data/finetune/train.jsonl \
    --val_file   data/finetune/val.jsonl   \
    --output_dir checkpoints/txgemma-9b-lipo

# Upload
python txgemma/finetune/upload_to_hub.py \
    --base_model   /scratch/ll5914/models/txgemma-9b-chat \
    --adapter_path checkpoints/txgemma-9b-lipo/final_adapter \
    --hf_repo      your_username/txgemma-9b-lipo-finetuned \
    --merged_dir   checkpoints/txgemma-9b-lipo-merged
```
