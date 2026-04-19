# LipoAgent: Coordinating Fine-Tuned LLM Agents for Safer Lipid Design

*ACL 2026*

This repository contains the official implementation of **LipoAgent**, a
safety-aware multi-agent LLM framework for lipid discovery that coordinates
fine-tuned **TxGemma** and **Qwen** agents to predict and verify lipid
nanoparticle (LNP) mRNA transfection efficiency for safer lipid design.

## Overview

**Figure 1** — Overview of LipoAgent, a safety-aware multi-agent LLM framework
for lipid discovery. ([`Figure1_final (1).pdf`](Figure1_final%20(1).pdf))

**Figure 2** — Overview of the LipoAgent framework. ([`figuer2_pipline (1).pdf`](figuer2_pipline%20(1).pdf))
- **(a) Fine-tuning and prompting pipeline** for constructing the predictor
  agent from a base LLM.
- **(b) Multi-agent collaboration in LipoAgent**, where agents coordinate with
  human feedback to iteratively filter and refine candidates toward
  high-efficiency lipids.

## Repository Structure

```
LipoAgent/
├── data/                       # Data preprocessing & datasets
│   ├── preprocess.py           # Excel → JSONL preprocessing
│   ├── virtual_library.xlsx    # Virtual lipid library (2500 molecules)
│   └── finetune/               # Training data preparation
│       └── build_dataset.py    # Build train/val splits from prediction labels
│
├── txgemma/                    # TxGemma-27B-Chat agents
│   ├── download.py             # Download TxGemma-27B weights
│   ├── predict_agent.py        # Predict Agent: efficiency score prediction
│   ├── verify_agent.py         # Verify Agent: two-pass re-scoring with detailed rationale
│   └── finetune/               # Finetuning scripts for TxGemma-9B
│       ├── download_9b.py      # Download TxGemma-9B weights
│       ├── prepare_data.py     # Convert labeled data → SFT format
│       ├── finetune_sft.py     # LoRA SFT finetuning
│       └── upload_to_hub.py    # Merge LoRA + push to HuggingFace Hub
│
├── qwen/                       # Qwen2.5-32B-Instruct agents
│   ├── predict_agent.py        # Predict Agent: efficiency score prediction
│   ├── verify_agent.py         # Verify Agent: molecular descriptor-based verification
│   └── finetune/               # Finetuning scripts for Qwen2.5
│       ├── finetune_sft.py     # LoRA SFT finetuning
│       └── upload_to_hub.py    # Merge LoRA + push to HuggingFace Hub
│
└── baseline/                   # Baseline comparisons
    ├── txgemma_9b/             # TxGemma-9B zero-shot baseline
    │   ├── download.py
    │   └── evaluate.py
    └── lantern/                # LANTERN fingerprint+MLP baseline
        ├── convert_data.py
        ├── evaluate_data.py
        ├── run_evaluation.py
        ├── within_2_analysis.py
        └── analyze_by_value_range.py
```

## Pipeline

### 1. Data Preprocessing
```bash
python data/preprocess.py
```
Converts `data/virtual_library.xlsx` → `data/virtual_library_preprocessed.jsonl`

### 2. TxGemma Agents

**Download weights (if not already present):**
```bash
python txgemma/download.py
```

**Predict Agent** — scores each molecule 1–10 with rationale:
```bash
python txgemma/predict_agent.py
```
Output: `data/txgemma_predict_results.json`

**Verify Agent** — two-pass re-scoring: fast score first, then detailed rationale for high/low scorers:
```bash
python txgemma/verify_agent.py
```
Output: `data/txgemma_verify_results.json`

### 3. Qwen Agents

**Predict Agent** — Qwen2.5-32B-Instruct efficiency prediction:
```bash
python qwen/predict_agent.py
```
Output: `data/qwen_predict_results.json`

**Verify Agent** — molecular descriptor analysis + Qwen reasoning-based verification:
```bash
python qwen/verify_agent.py
```
Output: `data/qwen_verify_results.json`

### 4. Finetuning

Prepare data, then run finetuning. See `txgemma/finetune/README.md` and `qwen/finetune/README.md` for full instructions.

```bash
# Step 1: build training data
python data/finetune/build_dataset.py

# Step 2: finetune TxGemma-9B
python txgemma/finetune/finetune_sft.py \
    --model_path google/txgemma-9b-chat \
    --train_file data/finetune/train.jsonl \
    --val_file   data/finetune/val.jsonl   \
    --output_dir checkpoints/txgemma-9b-lipo

# Step 3: finetune Qwen2.5-32B
python qwen/finetune/finetune_sft.py \
    --model_path Qwen/Qwen2.5-32B-Instruct \
    --train_file data/finetune/train.jsonl \
    --val_file   data/finetune/val.jsonl   \
    --output_dir checkpoints/qwen-32b-lipo
```

### 5. Baselines

**TxGemma-9B (zero-shot):**
```bash
python baseline/txgemma_9b/download.py   # one-time
python baseline/txgemma_9b/evaluate.py
```

**LANTERN:**
```bash
python baseline/lantern/convert_data.py
python baseline/lantern/evaluate_data.py
python baseline/lantern/run_evaluation.py
```

## Requirements

```bash
pip install torch transformers peft accelerate
pip install pandas openpyxl tqdm
pip install huggingface_hub scikit-learn scipy
```
