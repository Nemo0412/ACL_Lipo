#!/usr/bin/env bash
# Step 1 of 2: Fine-tune TxGemma-27B on 414 labelled lipids (LoRA SFT)
# Submit: sbatch data_0613/run_finetune_0613.sbatch
# Then:   sbatch data_0613/run_score_19k.sbatch   (after this job finishes)
#SBATCH -J txgemma_ft_0613
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH -t 24:00:00
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH -o /scratch/ll5914/logs/txgemma_ft_0613_%j.out
#SBATCH -e /scratch/ll5914/logs/txgemma_ft_0613_%j.err

set -euo pipefail

export HF_HOME="${HF_HOME:-/scratch/ll5914/huggingface}"
export TRANSFORMERS_VERBOSITY=warning
export TOKENIZERS_PARALLELISM=false

# Load HF token
if [[ -z "${HF_TOKEN:-}" ]] && [[ -f "${HOME}/.hf_token" ]]; then
  HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.hf_token")" && export HF_TOKEN
fi

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

cd "${HOME}/ACL_Lipo"

# Build dataset if not already done
if [[ ! -f data_0613/finetune/train_0613.jsonl ]]; then
  echo "Building fine-tune dataset..."
  python3 data_0613/build_finetune_0613.py
fi

MODEL="google/txgemma-27b-chat"
OUTPUT_DIR="checkpoints/txgemma-27b-lipo-0613"

echo "=== Fine-tuning TxGemma-27B (LoRA) ==="
echo "  Model      : ${MODEL}"
echo "  Train      : data_0613/finetune/train_0613.jsonl  (352 rows)"
echo "  Val        : data_0613/finetune/val_0613.jsonl    (62 rows)"
echo "  Output     : ${OUTPUT_DIR}"
echo "  GPUs       : ${SLURM_GPUS_ON_NODE:-2}"
echo "Started at $(date)"

torchrun --nproc_per_node=2 txgemma/finetune/finetune_sft.py \
  --model_path     "${MODEL}" \
  --train_file     "data_0613/finetune/train_0613.jsonl" \
  --val_file       "data_0613/finetune/val_0613.jsonl" \
  --output_dir     "${OUTPUT_DIR}" \
  --max_length     512 \
  --epochs         5 \
  --per_device_batch 1 \
  --grad_accum     8 \
  --lr             1e-4 \
  --lora_r         16 \
  --lora_alpha     32 \
  --save_steps     50 \
  --eval_steps     50 \
  --logging_steps  5

echo "Fine-tuning done at $(date)"
echo "Adapter saved to: ${OUTPUT_DIR}/final_adapter"
