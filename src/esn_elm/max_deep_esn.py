
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
