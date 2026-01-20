
import os
import sys
import numpy as np
import random
from sklearn.metrics import f1_score

# Add data_processing to path to import loader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
from chunked_data_loader import create_raw_sequences_chunked, CHUNK_SIZE, SEQ_LEN
from max_deep_esn import DeepESN

def adapter_batch_to_blocks(X, y):
    """
    Converts (N_samples, Seq_Len, N_Features) -> List of (Seq_Len, N_Features)
    DeepESN fit method expects list of blocks.
    """
    blocks = []
    for i in range(len(X)):
        # Each sample is treated as a short "block" sequence
        # Fix: Broadcast scalar label y[i] to vector of shape (Seq_Len,)
        T = X[i].shape[0]
        y_seq = np.full((T,), y[i])
        blocks.append((X[i], y_seq, None))
    return blocks

def main():
    base_dir = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev"
    train_csv = os.path.join(base_dir, "demos/Train.csv")
    
    if not os.path.exists(train_csv):
        print(f"Error: {train_csv} not found.")
        return

    print("Loading Real Data via Chunked Loader...")
    # Using existing loader to get the metadata features
    X_train, y_train, X_val, y_val = create_raw_sequences_chunked(
        train_csv, val_split=0.2, seq_len=SEQ_LEN, chunk_size=CHUNK_SIZE
    )
    
    print(f"Data Loaded: Train Shape {X_train.shape}, Val Shape {X_val.shape}")
    INPUT_DIM = X_train.shape[2]
    
    # Convert to DeepESN format
    train_blocks = adapter_batch_to_blocks(X_train, y_train)
    val_blocks = adapter_batch_to_blocks(X_val, y_val)
    
    # --- Tuning Loop ---
    print("\n--- Starting 10-Iteration Random Search on REAL DATA ---")
    best_score = -1.0
    best_config = None
    
    for i in range(10):
        config = {
            'n_layers': random.choice([2, 3, 4]), # Bias towards deep
            'res_dim': random.choice([500, 1000]), # Keep manageable for speed
            'spectral_radius': random.uniform(0.9, 1.1),
            'leak_rate': random.uniform(0.1, 0.4),
            'ridge_alpha': random.choice([0.1, 1.0, 5.0])
        }
        
        print(f"\n[Exp {i+1}/10] Config: {config}")
        
        try:
            model = DeepESN(
                input_dim=INPUT_DIM,
                n_layers=config['n_layers'],
                res_dim=config['res_dim'],
                spectral_radius=config['spectral_radius'],
                leak_rate=config['leak_rate'],
                ridge_alpha=config['ridge_alpha']
            )
            
            # Train
            # We suppress internal printing for cleaner tuning output
            model.fit(train_blocks, compute_metrics=False)
            
            # Validate
            # Flatten predictions for scoring
            y_true_all = []
            y_pred_all = []
            
            val_preds = model.predict(val_blocks)
            
            # val_preds is List of (Seq_Len,) - one prediction per Step in sequence?
            # ESN predict returns one prediction vector per step.
            # But wait: fit() trained on (T, ) labels? 
            # Check y_train shape from chunked loader: (N,) or (N, T)?
            # answer: (N,) usually in chunked_loader (label per sequence).
            # DeepESN expects sequences.
            
            # RE-ADJUSTMENT: 
            # create_raw_sequences_chunked returns y as (N,) - Class per sequence.
            # DeepESN expects y as (T,) - Class per timestep.
            # We must broaden y_train to (T,) for training DeepESN.
            
            # Simple Fix: Repeat label for all timesteps? 
            # Yes, if the sequence is short and represents one "state".
            
            # Updating fit logic inside this loop specifically:
            # Reconstruct blocks with (T,) labels
            
            train_blocks_seq = []
            for j in range(len(X_train)):
                T = X_train[j].shape[0]
                # Broadcast scalar label to vector
                y_seq = np.full((T,), y_train[j]) 
                train_blocks_seq.append((X_train[j], y_seq, None))
                
            val_blocks_seq = []
            for j in range(len(X_val)):
                T = X_val[j].shape[0]
                y_seq = np.full((T,), y_val[j])
                val_blocks_seq.append((X_val[j], y_seq, None))
            
            model.fit(train_blocks_seq, compute_metrics=False)
             
            # Predict
            preds = model.predict(val_blocks_seq)
            
            # Eval: Take the MEAN prediction or LAST prediction of the sequence to match the single label?
            # Let's take Majority Vote (Rounded Mean pattern) across the sequence
            final_preds = []
            for p_seq in preds:
                 # p_seq is shape (T,) continuous values
                 p_class = np.round(np.clip(p_seq, 0, 3)).astype(int)
                 # Majority vote
                 counts = np.bincount(p_class, minlength=4)
                 final_preds.append(np.argmax(counts))
            
            score = f1_score(y_val, final_preds, average='macro')
            print(f"Result: Val F1-Macro = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_config = config
                print(">>> NEW BEST!")
                
        except Exception as e:
            print(f"Failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*30)
    print(f"TUNING COMPLETE on Real Metadata.")
    print(f"Best F1: {best_score:.4f}")
    print(f"Best Config: {best_config}")
    print("="*30)

if __name__ == "__main__":
    main()
