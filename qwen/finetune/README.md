# Qwen Finetuning

This directory contains finetuning scripts for Qwen2.5-32B-Instruct agents.

## Predict Agent Finetuning

Fine-tune Qwen2.5-32B to improve efficiency score prediction on the
LNP virtual library. Uses the same JSONL format as TxGemma.

## Verify Agent Finetuning

Fine-tune Qwen to improve RDKit descriptor interpretation and rationale
generation for molecular verification.

## Training Data

See `data/preprocess.py` and `data/virtual_library.xlsx`.
Ground-truth test sets: `data/efficiency_test_data_rdkit.jsonl` and
`data/toxicity_test_data_rdkit.jsonl`.
