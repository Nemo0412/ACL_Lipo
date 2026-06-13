#!/usr/bin/env bash
# TxGemma-27B embedding-space two-pass agent
# Submit: sbatch txgemma/run_embedding_agent.sbatch
# Monitor: tail -f /scratch/ll5914/logs/tx_embed_<JOBID>.out
#SBATCH -J tx_embed_agent
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH -t 72:00:00
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH -o /scratch/ll5914/logs/tx_embed_%j.out
#SBATCH -e /scratch/ll5914/logs/tx_embed_%j.err

set -euo pipefail

export HF_HOME="${HF_HOME:-/scratch/ll5914/huggingface}"
export TRANSFORMERS_VERBOSITY=warning
export TOKENIZERS_PARALLELISM=false

# Load HF token from ~/.hf_token if not set
if [[ -z "${HF_TOKEN:-}" ]]; then
  [[ -f "${HOME}/.hf_token" ]] && HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.hf_token")" && export HF_TOKEN
fi
[[ -z "${HF_TOKEN:-}" ]] && { echo "ERROR: HF_TOKEN not set" >&2; exit 1; }

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD

cd "${HOME}/ACL_Lipo"

python txgemma/embedding_agent.py \
  --input  "${HOME}/ACL_Lipo/data/virtual_library_preprocessed.jsonl" \
  --output "${HOME}/ACL_Lipo/data/txgemma_embedding_results.json" \
  --checkpoint_every 50 \
  --verify_threshold 7

echo "Done at $(date)"
