#!/usr/bin/env bash
# TxGemma-27B: score all lipids in Smiles-Expension lipid.xlsx (long job ~1–2+ days for ~10k rows)
# Submit:  sbatch txgemma/run_predict_expansion.sbatch
# Monitor: tail -f /scratch/ll5914/logs/txgemma_expansion_<JOBID>.out
#SBATCH -J txgemma_expansion
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH -t 72:00:00
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH -o /scratch/ll5914/logs/txgemma_expansion_%j.out
#SBATCH -e /scratch/ll5914/logs/txgemma_expansion_%j.err

set -euo pipefail

export HF_HOME="${HF_HOME:-/scratch/ll5914/huggingface}"
export TRANSFORMERS_VERBOSITY=warning
export TOKENIZERS_PARALLELISM=false

# SLURM jobs often do not inherit your login-shell env. Hugging Face needs the token here.
# Prefer ONE of: (1) submit with  HF_TOKEN=hf_... sbatch ...  (2) write token to ~/.hf_token (chmod 600)
# (3) huggingface-cli login  → token is stored under ~/.cache/huggingface/token
if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f "${HOME}/.hf_token" ]]; then
    HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.hf_token")"
    export HF_TOKEN
  elif [[ -f "${HOME}/.cache/huggingface/token" ]]; then
    HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
    export HF_TOKEN
  fi
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set. Create ${HOME}/.hf_token, or run huggingface-cli login on this filesystem," >&2
  echo "       or submit with: HF_TOKEN=hf_... sbatch txgemma/run_predict_expansion.sbatch" >&2
  exit 1
fi

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

cd "${HOME}/ACL_Lipo"

# Fresh run (new path): old file is full of checkpoint IDs from failed inference;
# reusing the same path would skip all rows. Override with TXGEMMA_OUT if needed.
TXGEMMA_OUT="${TXGEMMA_OUT:-${HOME}/ACL_Lipo/data/smiles_expansion_tx27b_txgemma_rerun.json}"

echo "TxGemma output: ${TXGEMMA_OUT}"
echo "HF_HOME=${HF_HOME}"

python txgemma/predict_expansion_xlsx.py \
  --input "${HOME}/ACL_Lipo/Smiles-Expension lipid.xlsx" \
  --output "${TXGEMMA_OUT}" \
  --checkpoint_every 50

echo "Done at $(date)"
