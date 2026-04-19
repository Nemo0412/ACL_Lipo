#!/usr/bin/env python3
"""
Download google/txgemma-9b-chat from HuggingFace.

Requires HF_TOKEN env var or ~/.huggingface/token with access to the gated model.
Get access at: https://huggingface.co/google/txgemma-9b-chat

Usage:
  HF_TOKEN=hf_... python download_9b.py
  python download_9b.py --save_dir /scratch/myuser/models/txgemma-9b-chat
"""

import argparse
import os

from huggingface_hub import login, snapshot_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--save_dir",
        default="/scratch/ll5914/models/txgemma-9b-chat",
        help="Local directory to save the model",
    )
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        # Try standard cache location
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        token_file = os.path.join(hf_home, "token")
        if os.path.exists(token_file):
            with open(token_file) as f:
                token = f.read().strip()

    if not token:
        raise RuntimeError(
            "HuggingFace token not found.\n"
            "Set HF_TOKEN env var or run:  huggingface-cli login\n"
            "Then get access at: https://huggingface.co/google/txgemma-9b-chat"
        )

    login(token=token, add_to_git_credential=False)
    print(f"Downloading google/txgemma-9b-chat → {args.save_dir}")

    os.makedirs(args.save_dir, exist_ok=True)
    snapshot_download(
        repo_id="google/txgemma-9b-chat",
        local_dir=args.save_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=token,
    )

    print(f"\n✅ Downloaded to: {args.save_dir}")


if __name__ == "__main__":
    main()
