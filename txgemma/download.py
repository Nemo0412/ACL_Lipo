#!/usr/bin/env python3
"""
Download TxGemma-27B model from Hugging Face
"""

from huggingface_hub import snapshot_download
import os

# Local path for saved weights
model_name = "google/txgemma-27b-chat"
save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'Txgemma', 'txgemma-27b-chat')

print(f"Downloading model: {model_name}")
print(f"Save path: {save_path}")

# Create output directory
os.makedirs(save_path, exist_ok=True)

# Download snapshot
try:
    snapshot_download(
        repo_id=model_name,
        local_dir=save_path,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4
    )
    print(f"\nDownload complete. Saved to: {save_path}")
except Exception as e:
    print(f"Download error: {e}")
    print("\nIf you hit auth errors, run: huggingface-cli login")
