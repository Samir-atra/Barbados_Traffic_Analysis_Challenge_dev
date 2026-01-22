#!/usr/bin/env python3
"""DeepESN Hyperparameter Tuning with Real Train.csv Data.

This script connects the DeepESN model to the chunked data loader
to train on the actual traffic prediction dataset.
"""

import os
import sys
import numpy as np
import random
from sklearn.metrics import f1_score, accuracy_score, classification_report

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from esn_elm.max_deep_esn import DeepESN
from data_processing.chunked_data_loader import (
    create_raw_sequences_chunked,
    CHUNK_SIZE,
    SEQ_LEN
)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_CSV = os.path.join(BASE_DIR, "demos/Train.csv")
TRAIN_BALANCED_CSV = os.path.join(BASE_DIR, "demos/Train_Balanced_3k.csv")

# Use balanced dataset if available, otherwise full Train.csv
if os.path.exists(TRAIN_BALANCED_CSV):
    DATA_PATH = TRAIN_BALANCED_CSV
    print(f"Using balanced dataset: {TRAIN_BALANCED_CSV}")
else:
    DATA_PATH = TRAIN_CSV
    print(f"Using full dataset: {TRAIN_CSV}")


def prepare_blocks_from_sequences(X_seqs, y_labels, block_size=CHUNK_SIZE):
    """Converts sequence arrays into blocks format expected by DeepESN.
    
    The DeepESN expects blocks as List[(X_seq, y_seq, meta)] where:
        - X_seq: (T, input_dim) array
        - y_seq: (T,) array of labels
        - meta: Optional metadata (can be None)
    
    Args:
        X_seqs: NumPy array of shape (N, seq_len, n_features).
        y_labels: NumPy array of shape (N,).
        block_size: Number of sequences to group into each block.
        
    Returns:
        List of (X_block, y_block, None) tuples.
    """
    n_samples = len(X_seqs)
    n_features = X_seqs.shape[2]
    seq_len = X_seqs.shape[1]
    
    blocks = []
    
    # Group sequences into blocks
    for start_idx in range(0, n_samples, block_size):
        end_idx = min(start_idx + block_size, n_samples)
        
        # Stack sequences: each sequence becomes T timesteps
        # Block shape: (block_size * seq_len, n_features) approximately
        block_X_list = []
        block_y_list = []
        
        for i in range(start_idx, end_idx):
            # Each sequence is (seq_len, n_features)
            block_X_list.append(X_seqs[i])
            # Repeat the label for each timestep in sequence
            block_y_list.extend([y_labels[i]] * seq_len)
        
        # Stack into single block
        block_X = np.vstack(block_X_list)  # (T_total, n_features)
        block_y = np.array(block_y_list)    # (T_total,)
        
        blocks.append((block_X, block_y, None))
    
    return blocks


def objective_function(config, train_blocks, val_blocks, input_dim):
    """Trains and evaluates a DeepESN with the given config.
    
    Args:
        config: Dictionary of hyperparameters.
        train_blocks: Training data blocks.
        val_blocks: Validation data blocks.
        input_dim: Input feature dimension.
        
    Returns:
        F1-Macro score on validation set.
    """
    model = DeepESN(
        input_dim=input_dim,
        n_layers=config['n_layers'],
        res_dim=config['res_dim'],
        spectral_radius=config['spectral_radius'],
        leak_rate=config['leak_rate'],
        ridge_alpha=config['ridge_alpha'],
        random_state=42
    )
    
    # Train
    model.fit(train_blocks, compute_metrics=False)
    
    # Validate
    y_true = []
    y_pred_flat = []
    
    val_preds = model.predict(val_blocks)
    for i, p_seq in enumerate(val_preds):
        _, t_seq, _ = val_blocks[i]
        y_true.extend(t_seq)
        y_pred_flat.extend(p_seq)
    
    y_pred_class = np.round(np.clip(y_pred_flat, 0, 3)).astype(int)
    
    f1 = f1_score(y_true, y_pred_class, average='macro')
    acc = accuracy_score(y_true, y_pred_class)
    
    return f1, acc, y_true, y_pred_class


def main():
    """Main training and hyperparameter search function."""
    print("=" * 60)
    print("DeepESN Hyperparameter Tuning - Real Data")
    print("=" * 60)
    
    # Load data using chunked data loader
    print(f"\nLoading data from: {DATA_PATH}")
    X_train, y_train, X_val, y_val = create_raw_sequences_chunked(DATA_PATH)
    
    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  y_val:   {y_val.shape}")
    
    input_dim = X_train.shape[2]  # Number of features
    print(f"  Input dimension: {input_dim}")
    
    # Convert to blocks format for DeepESN
    print("\nConverting to DeepESN block format...")
    train_blocks = prepare_blocks_from_sequences(X_train, y_train, block_size=50)
    val_blocks = prepare_blocks_from_sequences(X_val, y_val, block_size=50)
    
    print(f"  Train blocks: {len(train_blocks)}")
    print(f"  Val blocks:   {len(val_blocks)}")
    
    # Random Search for Hyperparameters
    print("\n" + "=" * 60)
    print("Starting Random Search (10 Experiments)")
    print("=" * 60)
    
    best_score = -1.0
    best_config = None
    best_results = None
    
    search_space = {
        'n_layers': [1, 2, 3, 4],
        'res_dim': [200, 500, 800, 1000],
        'spectral_radius': (0.8, 1.2),  # Uniform range
        'leak_rate': (0.05, 0.5),        # Uniform range
        'ridge_alpha': [0.01, 0.1, 1.0, 10.0]
    }
    
    for i in range(10):
        # Sample hyperparameters
        config = {
            'n_layers': random.choice(search_space['n_layers']),
            'res_dim': random.choice(search_space['res_dim']),
            'spectral_radius': random.uniform(*search_space['spectral_radius']),
            'leak_rate': random.uniform(*search_space['leak_rate']),
            'ridge_alpha': random.choice(search_space['ridge_alpha'])
        }
        
        print(f"\n--- Experiment {i+1}/10 ---")
        print(f"Config: layers={config['n_layers']}, res={config['res_dim']}, "
              f"rho={config['spectral_radius']:.3f}, leak={config['leak_rate']:.3f}, "
              f"alpha={config['ridge_alpha']}")
        
        try:
            f1, acc, y_true, y_pred = objective_function(
                config, train_blocks, val_blocks, input_dim)
            
            print(f"Result: F1-Macro={f1:.4f}, Accuracy={acc:.4f}")
            
            if f1 > best_score:
                best_score = f1
                best_config = config.copy()
                best_results = (y_true, y_pred)
                print(">>> NEW BEST!")
                
        except Exception as e:
            print(f"Experiment failed: {e}")
    
    # Final Report
    print("\n" + "=" * 60)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 60)
    print(f"\nBest F1-Macro: {best_score:.4f}")
    print(f"\nBest Configuration:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    
    if best_results:
        print("\nClassification Report (Best Model):")
        y_true, y_pred = best_results
        labels = ['free flowing', 'light delay', 'moderate delay', 'heavy delay']
        print(classification_report(y_true, y_pred, target_names=labels))


if __name__ == "__main__":
    main()
