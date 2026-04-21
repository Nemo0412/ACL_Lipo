#!/usr/bin/env python3
"""
LoRA SFT finetuning of Qwen2.5-32B-Instruct for LNP efficiency prediction.

Uses PEFT LoRA + HuggingFace Trainer. Same training data format as TxGemma.

Usage (single GPU):
  python finetune_sft.py \
      --model_path Qwen/Qwen2.5-32B-Instruct \
      --train_file data/finetune/train.jsonl \
      --val_file   data/finetune/val.jsonl   \
      --output_dir checkpoints/qwen-32b-lipo
"""

import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def apply_chat_template_and_tokenize(examples, tokenizer, max_length: int):
    """
    Tokenize full conversation; mask system+user tokens with -100 so the
    model only learns to generate the assistant turn.
    """
    all_input_ids = []
    all_labels    = []

    for messages in examples["messages"]:
        # Full conversation
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        full_ids = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )["input_ids"]

        # Prefix (system + user), ends with generation prompt marker
        prefix_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix_ids = tokenizer(
            prefix_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )["input_ids"]

        n_prefix = len(prefix_ids)
        labels   = [-100] * n_prefix + full_ids[n_prefix:]

        full_ids = full_ids[:max_length]
        labels   = labels[:max_length]
        assert len(full_ids) == len(labels)

        all_input_ids.append(full_ids)
        all_labels.append(labels)

    return {"input_ids": all_input_ids, "labels": all_labels}


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",   default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--train_file",   required=True)
    p.add_argument("--val_file",     required=True)
    p.add_argument("--output_dir",   default="checkpoints/qwen-32b-lipo")
    p.add_argument("--max_length",   type=int,   default=1024)

    # LoRA
    p.add_argument("--lora_r",       type=int,   default=16)
    p.add_argument("--lora_alpha",   type=int,   default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Training
    p.add_argument("--epochs",          type=int,   default=3)
    p.add_argument("--per_device_batch",type=int,   default=1)
    p.add_argument("--grad_accum",      type=int,   default=8)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--warmup_ratio",    type=float, default=0.05)
    p.add_argument("--weight_decay",    type=float, default=0.01)
    p.add_argument("--save_steps",      type=int,   default=100)
    p.add_argument("--eval_steps",      type=int,   default=100)
    p.add_argument("--logging_steps",   type=int,   default=10)
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main    = local_rank == 0

    if is_main:
        print(f"\n{'='*70}")
        print("Qwen2.5 LoRA SFT Finetuning")
        print(f"{'='*70}")
        print(f"  Model      : {args.model_path}")
        print(f"  Train data : {args.train_file}")
        print(f"  Val data   : {args.val_file}")
        print(f"  Output dir : {args.output_dir}")
        print(f"  LoRA r/α   : {args.lora_r}/{args.lora_alpha}")
        print(f"  Max length : {args.max_length}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    if is_main:
        print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Base model ────────────────────────────────────────────────────────────
    if is_main:
        print("Loading base model (bf16)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        # Qwen2.5 attention + FFN projection layers
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()   # required when gradient_checkpointing=True with LoRA
    if is_main:
        model.print_trainable_parameters()

    # ── Datasets ──────────────────────────────────────────────────────────────
    if is_main:
        print("\nPreparing datasets...")

    def prepare_split(path):
        raw = load_jsonl(path)
        ds  = Dataset.from_list(raw)
        ds  = ds.map(
            apply_chat_template_and_tokenize,
            fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
            batched=True,
            remove_columns=ds.column_names,
            desc=f"Tokenizing {os.path.basename(path)}",
        )
        return ds

    train_ds = prepare_split(args.train_file)
    val_ds   = prepare_split(args.val_file)

    if is_main:
        print(f"  Train examples: {len(train_ds)}")
        print(f"  Val   examples: {len(val_ds)}")

    # ── Collator ──────────────────────────────────────────────────────────────
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    # ── TrainingArguments ─────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    if is_main:
        print("\nStarting training...")

    trainer.train()

    if is_main:
        print("\nSaving final LoRA adapter...")
        model.save_pretrained(os.path.join(args.output_dir, "final_adapter"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "final_adapter"))
        print(f"Adapter saved to: {args.output_dir}/final_adapter")


if __name__ == "__main__":
    main()
