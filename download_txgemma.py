#!/usr/bin/env python3
"""
Download TxGemma-27B model from Hugging Face
"""

from huggingface_hub import snapshot_download
import os

# 设置模型保存路径
model_name = "google/txgemma-27b-chat"
save_path = "/mnt/3fs/dots-pretrain/leshu/workspace/Txgemma/txgemma-27b-chat"

print(f"开始下载模型: {model_name}")
print(f"保存路径: {save_path}")

# 创建保存目录
os.makedirs(save_path, exist_ok=True)

# 下载模型
try:
    snapshot_download(
        repo_id=model_name,
        local_dir=save_path,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4
    )
    print(f"\n模型下载完成！保存在: {save_path}")
except Exception as e:
    print(f"下载出错: {e}")
    print("\n如果遇到认证问题，请运行: huggingface-cli login")
