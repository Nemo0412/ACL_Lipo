#!/usr/bin/env bash
# Step 2 of 2: Score 19k screening lipids using fine-tuned TxGemma-27B
# Run AFTER run_finetune_0613.sbatch completes.
# Submit: sbatch data_0613/run_score_19k.sbatch
#   OR chain automatically:  sbatch --dependency=afterok:<FT_JOBID> data_0613/run_score_19k.sbatch
#SBATCH -J txgemma_score_19k
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH -o /scratch/ll5914/logs/txgemma_score_19k_%j.out
#SBATCH -e /scratch/ll5914/logs/txgemma_score_19k_%j.err

set -euo pipefail

export HF_HOME="${HF_HOME:-/scratch/ll5914/huggingface}"
export TRANSFORMERS_VERBOSITY=warning
export TOKENIZERS_PARALLELISM=false

if [[ -z "${HF_TOKEN:-}" ]] && [[ -f "${HOME}/.hf_token" ]]; then
  HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.hf_token")" && export HF_TOKEN
fi

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

cd "${HOME}/ACL_Lipo"

ADAPTER="${HOME}/ACL_Lipo/checkpoints/txgemma-27b-lipo-0613/final_adapter"
INPUT="${HOME}/ACL_Lipo/data_0613/Smiles for lipid screening-19000.xlsx"
OUTPUT="${HOME}/ACL_Lipo/data_0613/scored_19k.json"

echo "=== Scoring 19k lipids with fine-tuned TxGemma-27B ==="
echo "  Adapter : ${ADAPTER}"
echo "  Input   : ${INPUT}"
echo "  Output  : ${OUTPUT}"
echo "Started at $(date)"

python3 data_0613/score_19k.py \
  --adapter "${ADAPTER}" \
  --base_model "google/txgemma-27b-chat" \
  --input  "${INPUT}" \
  --output "${OUTPUT}" \
  --checkpoint_every 100

echo "Done at $(date)"
echo "Results: ${OUTPUT}"
echo "CSV:     ${OUTPUT/.json/.csv}"
