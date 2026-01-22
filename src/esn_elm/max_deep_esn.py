
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin

class DeepESN(BaseEstimator, ClassifierMixin):
    """
    Deep Echo State Network (DeepESN) implementation.
    Consists of a stack of reservoir layers. Each layer feeds into the next.
    The final state used for prediction is the concatenation of states from all layers.
    """
    def __init__(self, input_dim=1280, n_layers=2, res_dim=1000, spectral_radius=0.9, leak_rate=0.2, ridge_alpha=1.0, random_state=42):
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.res_dim = res_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state
        
        # Initialize Architecture
        self.layers = [] # List of dicts {'W_in', 'W_res'}
        rng = np.random.RandomState(self.random_state)
        
        for i in range(n_layers):
            # Input dimension for layer i: 
            # Layer 0 takes original input (input_dim)
            # Layer >0 takes state of previous layer (res_dim)
            curr_input_dim = input_dim if i == 0 else res_dim
            
            # Input weights
            W_in = rng.uniform(-1, 1, (res_dim, curr_input_dim))
            
            # Reservoir weights (Sparse)
            W_res = rng.uniform(-1, 1, (res_dim, res_dim))
            mask = rng.rand(res_dim, res_dim) > 0.95
            W_res[mask] = 0
            
            # Spectral Radius Scaling
            try:
                eigenvalues = np.linalg.eigvals(W_res)
                max_eig = np.max(np.abs(eigenvalues))
                if max_eig > 0:
                    W_res *= (self.spectral_radius / max_eig)
            except:
                W_res *= 0.9 # Fallback
                
            self.layers.append({
                'W_in': W_in,
                'W_res': W_res
            })
            
        self.readout = Ridge(alpha=self.ridge_alpha)
        
    def get_states_sequence(self, input_seq):
        """
        Processes a sequence (T, input_dim) through the Deep ESN stack.
        Returns: (T, n_layers * res_dim) - Concatenated states of all layers
        """
        T = input_seq.shape[0]
        
        # We need to store full sequence of states for each layer to feed to the next
        # layer_states: prediction input for next layer
        prev_layer_seq = input_seq # Start with actual input
        
        all_layers_collected_states = [] # To be concatenated for readout
        
        for i, layer in enumerate(self.layers):
            W_in = layer['W_in']
            W_res = layer['W_res']
            
            current_layer_states = np.zeros((T, self.res_dim))
            x = np.zeros(self.res_dim)
            
            for t in range(T):
                u = prev_layer_seq[t]
                
                # Standard ESN Equestion: x(t) = (1-a)x(t-1) + a*tanh(Win*u + Wres*x(t-1))
                pre = np.dot(W_in, u) + np.dot(W_res, x)
                update = np.tanh(pre)
                x = (1 - self.leak_rate) * x + self.leak_rate * update
                
                current_layer_states[t] = x
            
            all_layers_collected_states.append(current_layer_states)
            prev_layer_seq = current_layer_states # Output of this layer is input to next
            
        # Concatenate all layers: (T, n_layers * res_dim)
        final_states = np.hstack(all_layers_collected_states)
        return final_states

    def fit(self, blocks, compute_metrics=True):
        """
        Trains the execution readout on sequential blocks.
        blocks: List of (X_seq, y_seq, ...)
        """
        all_states = []
        all_targets = []
        
        print(f"DeepESN: Training on {len(blocks)} blocks (Layers={self.n_layers}, ResDim={self.res_dim})...")
        
        for idx, (X_seq, y_seq, _) in enumerate(blocks):
            states_seq = self.get_states_sequence(X_seq)
            all_states.append(states_seq)
            all_targets.append(y_seq)
            
        # Stack
        X_train_res = np.vstack(all_states)
        y_train_flat = np.concatenate(all_targets)
        
        self.readout.fit(X_train_res, y_train_flat)
        
        if compute_metrics:
            y_pred = self.readout.predict(X_train_res)
            y_pred_class = np.round(np.clip(y_pred, 0, 3)).astype(int)
            acc = accuracy_score(y_train_flat, y_pred_class)
            f1 = f1_score(y_train_flat, y_pred_class, average='macro')
            print(f"[TRAIN] DeepESN Accuracy: {acc:.4f} | F1-Macro: {f1:.4f}")
            
    def predict(self, blocks):
        all_preds = []
        for X_seq, _, _ in blocks:
            states_seq = self.get_states_sequence(X_seq)
            preds_seq = self.readout.predict(states_seq)
            all_preds.append(preds_seq)
        return all_preds

# --- Training with Real Data from Chunked Data Loader ---
if __name__ == "__main__":
    import os
    import sys
    from sklearn.metrics import classification_report
    
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data_processing.chunked_data_loader import (
        create_raw_sequences_chunked,
        SEQ_LEN
    )
    
    # Configuration
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    TRAIN_CSV = os.path.join(BASE_DIR, "demos/Train.csv")
    TRAIN_BALANCED_CSV = os.path.join(BASE_DIR, "demos/Train_Balanced_3k.csv")
    
    # Use balanced dataset if available
    DATA_PATH = TRAIN_BALANCED_CSV if os.path.exists(TRAIN_BALANCED_CSV) else TRAIN_CSV
    
    def prepare_blocks(X_seqs, y_labels, block_size=50):
        """Converts sequences to DeepESN block format.
        
        Args:
            X_seqs: (N, seq_len, n_features) array.
            y_labels: (N,) array.
            block_size: Sequences per block.
            
        Returns:
            List of (X_block, y_block, None) tuples.
        """
        n_samples = len(X_seqs)
        seq_len = X_seqs.shape[1]
        blocks = []
        
        for start in range(0, n_samples, block_size):
            end = min(start + block_size, n_samples)
            
            block_X = []
            block_y = []
            for i in range(start, end):
                block_X.append(X_seqs[i])
                block_y.extend([y_labels[i]] * seq_len)
            
            blocks.append((np.vstack(block_X), np.array(block_y), None))
        
        return blocks
    
    print("=" * 60)
    print("DeepESN Training on Real Traffic Data")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from: {DATA_PATH}")
    X_train, y_train, X_val, y_val = create_raw_sequences_chunked(DATA_PATH)
    
    input_dim = X_train.shape[2]
    print(f"\nData loaded:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"  Input dimension: {input_dim}")
    
    # Convert to blocks
    train_blocks = prepare_blocks(X_train, y_train)
    val_blocks = prepare_blocks(X_val, y_val)
    print(f"  Train blocks: {len(train_blocks)}, Val blocks: {len(val_blocks)}")
    
    # Train DeepESN
    print("\n" + "=" * 60)
    print("Training DeepESN")
    print("=" * 60)
    
    esn = DeepESN(
        input_dim=input_dim,
        n_layers=2,
        res_dim=500,
        spectral_radius=0.95,
        leak_rate=0.2,
        ridge_alpha=1.0,
        random_state=42
    )
    
    esn.fit(train_blocks, compute_metrics=True)
    
    # Evaluate on validation set
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    
    val_preds = esn.predict(val_blocks)
    
    y_true = []
    y_pred_flat = []
    for i, preds in enumerate(val_preds):
        _, y_seq, _ = val_blocks[i]
        y_true.extend(y_seq)
        y_pred_flat.extend(preds)
    
    y_pred_class = np.round(np.clip(y_pred_flat, 0, 3)).astype(int)
    
    acc = accuracy_score(y_true, y_pred_class)
    f1 = f1_score(y_true, y_pred_class, average='macro')
    
    print(f"\nValidation Accuracy: {acc:.4f}")
    print(f"Validation F1-Macro: {f1:.4f}")
    
    print("\nClassification Report:")
    labels = ['free flowing', 'light delay', 'moderate delay', 'heavy delay']
    print(classification_report(y_true, y_pred_class, target_names=labels))
    
    # ============================================================
    # SUBMISSION FILE GENERATION
    # ============================================================
    print("\n" + "=" * 60)
    print("Generating Submission File")
    print("=" * 60)
    
    TEST_CSV = os.path.join(BASE_DIR, "demos/TestInputSegments.csv")
    SAMPLE_SUB = os.path.join(BASE_DIR, "demos/SampleSubmission.csv")
    OUTPUT_CSV = os.path.join(BASE_DIR, "demos/submission.csv")
    
    if not os.path.exists(TEST_CSV):
        print(f"Test file not found: {TEST_CSV}")
    else:
        import pandas as pd
        from data_processing.chunked_data_loader import get_features_and_labels
        
        # Load test data and sample submission
        test_df = pd.read_csv(TEST_CSV)
        sample_sub = pd.read_csv(SAMPLE_SUB)
        
        print(f"Test data: {len(test_df)} rows")
        print(f"Sample submission: {len(sample_sub)} IDs to predict")
        
        # Extract features from test data (no labels)
        test_features, _ = get_features_and_labels(test_df)
        
        # Create test blocks - group by cycle_phase for sequential processing
        congestion_map_reverse = {0: 'free flowing', 1: 'light delay', 
                                   2: 'moderate delay', 3: 'heavy delay'}
        
        # Build a mapping from segment ID to prediction
        predictions_map = {}
        
        # Process test data by groups (cycle_phase)
        for cycle_phase, group in test_df.groupby('cycle_phase'):
            group = group.sort_values('time_segment_id')
            indices = group.index.tolist()
            
            if len(indices) < SEQ_LEN + 1:
                # Too short for sequence, use last available features
                group_features = test_features[indices]
                if len(group_features) > 0:
                    # Pad to minimum sequence length
                    while len(group_features) < SEQ_LEN:
                        group_features = np.vstack([group_features, group_features[-1:]])
            else:
                group_features = test_features[indices]
            
            # Process through ESN
            if len(group_features) >= SEQ_LEN:
                states = esn.get_states_sequence(group_features)
                preds = esn.readout.predict(states)
                pred_classes = np.round(np.clip(preds, 0, 3)).astype(int)
                
                # Map predictions back to segment IDs
                for j, idx in enumerate(indices):
                    if j < len(pred_classes):
                        row = test_df.loc[idx]
                        pred_label = congestion_map_reverse[pred_classes[j]]
                        
                        # Store predictions for both enter and exit
                        predictions_map[row['ID_enter']] = pred_label
                        predictions_map[row['ID_exit']] = pred_label
        
        # Create submission dataframe
        submission_rows = []
        for _, row in sample_sub.iterrows():
            seg_id = row['ID']
            if seg_id in predictions_map:
                pred = predictions_map[seg_id]
            else:
                # Default prediction if not found
                pred = 'free flowing'
            
            submission_rows.append({
                'ID': seg_id,
                'Target': pred,
                'Target_Accuracy': pred
            })
        
        submission_df = pd.DataFrame(submission_rows)
        submission_df.to_csv(OUTPUT_CSV, index=False)
        
        print(f"\nSubmission file saved to: {OUTPUT_CSV}")
        print(f"Total predictions: {len(submission_df)}")
        print("\nPrediction distribution:")
        print(submission_df['Target'].value_counts())
