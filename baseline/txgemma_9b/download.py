#!/usr/bin/env python3
"""
下载 Google TXGemma-9B 模型到本地
"""

from huggingface_hub import snapshot_download
import os

model_name = "google/txgemma-9b-chat"
local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "txgemma-9b-chat")

print(f"Downloading {model_name} to {local_dir}...")
print("This may take a while depending on your network speed...")

# 创建目录
os.makedirs(local_dir, exist_ok=True)

# 下载模型
snapshot_download(
    repo_id=model_name,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True
)

print(f"\n✅ Model downloaded successfully to: {local_dir}")
