#!/usr/bin/env python3
"""
Evaluate LANTERN on Qwen data (efficiency and toxicity)
"""
import os
import sys
import pandas as pd
import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Import LANTERN's featurizers
from deepchem.feat import CircularFingerprint, RDKitDescriptors

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

print("=" * 80)
print("Extracting fingerprints for Qwen test data...")
print("=" * 80)

# Load data
efficiency_df = pd.read_csv(os.path.join(DATA_DIR, 'efficiency_test_qwen.csv'))
toxicity_df = pd.read_csv(os.path.join(DATA_DIR, 'toxicity_test_qwen.csv'))

print(f"\nEfficiency test samples: {len(efficiency_df)}")
print(f"Toxicity test samples: {len(toxicity_df)}")

# Extract fingerprints for efficiency data
print("\n" + "=" * 80)
print("Extracting Circular Fingerprints for efficiency data...")
print("=" * 80)

circular_featurizer = CircularFingerprint(size=2048, chiral=True)
efficiency_smiles = efficiency_df['SMILES'].tolist()
efficiency_circular_fps = circular_featurizer.featurize(efficiency_smiles)
print(f"Circular fingerprint shape: {efficiency_circular_fps[0].shape}")

print("\n" + "=" * 80)
print("Extracting Expert Fingerprints for efficiency data...")
print("=" * 80)

expert_featurizer = RDKitDescriptors()
efficiency_expert_fps = expert_featurizer.featurize(efficiency_smiles)
print(f"Expert fingerprint shape: {efficiency_expert_fps[0].shape}")

# Truncate expert fingerprints to 210 dimensions to match LANTERN training data
efficiency_expert_fps_210 = [fp[:210] for fp in efficiency_expert_fps]
print(f"Truncated expert fingerprint shape: {efficiency_expert_fps_210[0].shape}")

# Save efficiency fingerprints
eff_fp_dir = os.path.join(DATA_DIR, 'fingerprints', 'efficiency_test_qwen')
os.makedirs(eff_fp_dir, exist_ok=True)

efficiency_circular_dict = {smiles: fp for smiles, fp in zip(efficiency_smiles, efficiency_circular_fps)}
efficiency_expert_dict = {smiles: fp for smiles, fp in zip(efficiency_smiles, efficiency_expert_fps_210)}

import pickle
with open(os.path.join(eff_fp_dir, 'circular.pkl'), 'wb') as f:
    pickle.dump(efficiency_circular_dict, f)
print("Saved: efficiency_test_qwen/circular.pkl")

with open(os.path.join(eff_fp_dir, 'expert.pkl'), 'wb') as f:
    pickle.dump(efficiency_expert_dict, f)
print("Saved: efficiency_test_qwen/expert.pkl")

# Extract fingerprints for toxicity data
print("\n" + "=" * 80)
print("Extracting fingerprints for toxicity data...")
print("=" * 80)

toxicity_smiles = toxicity_df['SMILES'].tolist()
toxicity_circular_fps = circular_featurizer.featurize(toxicity_smiles)
toxicity_expert_fps = expert_featurizer.featurize(toxicity_smiles)

print(f"Circular fingerprint shape: {toxicity_circular_fps[0].shape}")
print(f"Expert fingerprint shape: {toxicity_expert_fps[0].shape}")

# Truncate expert fingerprints to 210 dimensions
toxicity_expert_fps_210 = [fp[:210] for fp in toxicity_expert_fps]

# Save toxicity fingerprints
tox_fp_dir = os.path.join(DATA_DIR, 'fingerprints', 'toxicity_test_qwen')
os.makedirs(tox_fp_dir, exist_ok=True)

toxicity_circular_dict = {smiles: fp for smiles, fp in zip(toxicity_smiles, toxicity_circular_fps)}
toxicity_expert_dict = {smiles: fp for smiles, fp in zip(toxicity_smiles, toxicity_expert_fps_210)}

with open(os.path.join(tox_fp_dir, 'circular.pkl'), 'wb') as f:
    pickle.dump(toxicity_circular_dict, f)
print("Saved: toxicity_test_qwen/circular.pkl")

with open(os.path.join(tox_fp_dir, 'expert.pkl'), 'wb') as f:
    pickle.dump(toxicity_expert_dict, f)
print("Saved: toxicity_test_qwen/expert.pkl")

print("\n" + "=" * 80)
print("Fingerprint extraction complete!")
print("=" * 80)
