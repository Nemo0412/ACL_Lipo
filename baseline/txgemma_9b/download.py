#!/usr/bin/env python3
"""
Download Google TxGemma-9B-Chat weights to a local directory.
"""

from huggingface_hub import snapshot_download
import os

model_name = "google/txgemma-9b-chat"
local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "txgemma-9b-chat")

print(f"Downloading {model_name} to {local_dir}...")
print("This may take a while depending on your network speed...")

# Create output directory
os.makedirs(local_dir, exist_ok=True)

# Download snapshot
snapshot_download(
    repo_id=model_name,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True
)

print(f"\n✅ Model downloaded successfully to: {local_dir}")
