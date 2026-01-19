"""Echo State Network (ESN) for Traffic Prediction.

This script implements an Echo State Network (ESN), a type of Reservoir Computing
system designed specifically for time-series data. It works by having a large,
sparsely connected, random fixed "reservoir" of neurons.

Training involves:
1. Feeding the input sequence into the reservoir.
2. Collecting the reservoir states.
3. Solving a simple linear regression to map states to output labels.

NO backpropagation through time (BPTT) is used.

Workflow:
1. Load Train.csv and format into sequential blocks (N, 15, Features).
2. Run sequences through ESN Reservoir.
3. Train Readout Layer (Ridge Regression).
4. Predict on Test Set.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from scipy import sparse

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)

def get_features_and_labels(df):
    """Extracts features and labels."""
    df = df.copy()
    df['video_time'] = pd.to_datetime(df['video_time'])
    df['hour'] = df['video_time'].dt.hour / 23.0
    df['minute'] = df['video_time'].dt.minute / 59.0
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek / 6.0
    
    view_map = {
        'Norman Niles #1': 0, 
        'Norman Niles #2': 1, 
        'Norman Niles #3': 2, 
        'Norman Niles #4': 3
    }
    df['view_id'] = df['view_label'].map(view_map)
    df['seg_id_norm'] = df['time_segment_id'] / 5000.0
    
    congestion_map = {
        'free flowing': 0,
        'light delay': 1,
        'moderate delay': 2,
        'heavy delay': 3
    }
    if 'congestion_enter_rating' in df.columns:
        df['enter_id'] = df['congestion_enter_rating'].map(congestion_map).fillna(0).astype(int)
        labels = df['enter_id'].values
    else:
        labels = None
    
    view_1hot = pd.get_dummies(df['view_id'], prefix='view').reindex(
        columns=['view_0', 'view_1', 'view_2', 'view_3'], fill_value=0).astype(float).values
    
    if 'signaling' in df.columns:
        sig_map = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
        df['sig_id'] = df['signaling'].map(sig_map).fillna(0)
    else:
        df['sig_id'] = 0
    
    sig_1hot = pd.get_dummies(df['sig_id'], prefix='sig').reindex(
        columns=['sig_0', 'sig_1', 'sig_2', 'sig_3'], fill_value=0).astype(float).values
    
    features = np.concatenate([
        df[['hour', 'minute', 'day_of_week', 'seg_id_norm', 'view_id']].values,
        view_1hot,
        sig_1hot
    ], axis=1).astype('float32') # 13 features

    return features, labels

def identify_blocks(group):
    group = group.sort_values('time_segment_id')
    ids = group['time_segment_id'].values
    is_break = np.zeros(len(ids), dtype=int)
    is_break[1:] = (ids[1:] != ids[:-1] + 1).astype(int)
    group['block_id'] = np.cumsum(is_break)
    return group

def create_raw_sequences(csv_path, val_split=0.2, seq_len=15):
    """Processes CSV into 3D sequential samples (N, Time, Feat)."""
    df = pd.read_csv(csv_path)
    train_X, train_y = [], []
    val_X, val_y = [], []
    
    print(f"Loading {len(df)} rows from {csv_path}...")
    for label, group in df.groupby('view_label'):
        group = identify_blocks(group)
        
        for b_id, block in group.groupby('block_id'):
            if len(block) < seq_len + 1: continue
            
            # Split block first (Anti-Leakage)
            n_block = len(block)
            n_train_rows = int(n_block * (1 - val_split))
            
            train_block = block.iloc[:n_train_rows]
            val_block = block.iloc[n_train_rows:]
            
            def make_seqs(sub_block):
                feats, labels = get_features_and_labels(sub_block)
                X_w, y_w = [], []
                for i in range(len(feats) - seq_len):
                    # KEEP 2D Structure: (15, Features)
                    X_w.append(feats[i : i + seq_len])
                    y_w.append(labels[i + seq_len])
                return X_w, y_w
            
            if len(train_block) >= seq_len + 1:
                tx, ty = make_seqs(train_block)
                train_X.extend(tx)
                train_y.extend(ty)
            
            if len(val_block) >= seq_len + 1:
                vx, vy = make_seqs(val_block)
                val_X.extend(vx)
                val_y.extend(vy)
                
    return np.array(train_X), np.array(train_y), np.array(val_X), np.array(val_y)

class ESNClassifier:
    """Echo State Network for Sequence Classification."""
    
    def __init__(self, input_dim, reservoir_dim=500, spectral_radius=0.9, 
                 leak_rate=0.3, sparsity=0.1, alpha=1.0, seed=42):
        """
        Args:
            input_dim: Number of input features.
            reservoir_dim: Neurons in reservoir.
            spectral_radius: Scaling of reservoir weight matrix (rho).
            leak_rate: Leaking rate (alpha) in state update.
            sparsity: Probability of connection in reservoir.
            alpha: Ridge regression regularization.
        """
        self.reservoir_dim = reservoir_dim
        self.leak_rate = leak_rate
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        
        # 1. Input Weights (Dense, Random)
        # Uniform [-0.5, 0.5]
        self.W_in = (self.rng.random((reservoir_dim, input_dim)) - 0.5)
        
        # 2. Reservoir Weights (Sparse, Random)
        # Create sparse random matrix
        W = sparse.random(reservoir_dim, reservoir_dim, density=sparsity, random_state=seed)
        # Get eigenvalues to scale spectral radius
        eigenvalues = np.linalg.eigvals(W.toarray())
        max_eigen = np.max(np.abs(eigenvalues))
        self.W_res = W * (spectral_radius / max_eigen)
        
        self.W_out = None
        self.scaler = StandardScaler()
        
    def _get_states(self, X):
        """Run reservoir dynamics on batch of sequences."""
        n_samples, n_steps, _ = X.shape
        
        # Initialize states
        # Can be done sample-by-sample or batched if memory allows. 
        # For simplicity and speed in Python, we do loop over time steps for the whole batch.
        
        states = np.zeros((n_samples, self.reservoir_dim))
        
        # X is (N, T, F)
        # W_in is (Res, F) -> X @ W_in.T -> (N, Res)
        # W_res is (Res, Res) (sparse)
        
        for t in range(n_steps):
            u_t = X[:, t, :] # (N, F)
            
            # Input contribution
            input_part = np.dot(u_t, self.W_in.T) # (N, Res)
            
            # Recurrent contribution
            # W_res is sparse, so use safe sparse dot or dense dot
            res_part = self.W_res.dot(states.T).T # (N, Res)
            
            # Update state
            # x_t = (1-a)x_{t-1} + a * tanh(Win*u + Wres*x)
            update = np.tanh(input_part + res_part)
            states = (1 - self.leak_rate) * states + self.leak_rate * update
            
        return states # Return state after LAST time step for classification
    
    def fit(self, X, y):
        # Flatten time and features for scaling (ignoring sequence nature for scaling stats)
        # Actually scaling per feature across all (N*T) is best
        N, T, F = X.shape
        X_flat = X.reshape(-1, F)
        self.scaler.fit(X_flat)
        X_scaled = self.scaler.transform(X_flat).reshape(N, T, F)
        
        print(f"Running Reservoir for {N} sequences...")
        states = self._get_states(X_scaled)
        
        # Add bias to states for readout
        states_bias = np.hstack([states, np.ones((N, 1))])
        
        # Targets
        n_classes = 4
        Y_onehot = np.eye(n_classes)[y]
        
        # Ridge Regression
        # W_out = (S^T S + alpha*I)^-1 S^T Y
        res_size = self.reservoir_dim + 1
        I = np.eye(res_size)
        
        print(f"Solving Readout ({res_size} dims)...")
        A = np.dot(states_bias.T, states_bias) + self.alpha * I
        B = np.dot(states_bias.T, Y_onehot)
        
        self.W_out = np.linalg.solve(A, B)
        print("ESN Training Complete.")
        
    def predict(self, X):
        N, T, F = X.shape
        X_scaled = self.scaler.transform(X.reshape(-1, F)).reshape(N, T, F)
        states = self._get_states(X_scaled)
        states_bias = np.hstack([states, np.ones((N, 1))])
        
        Y_pred = np.dot(states_bias, self.W_out)
        return np.argmax(Y_pred, axis=1)

def main():
    base = "/teamspace/studios/this_studio/Barbados_Traffic_Analysis_Challenge_dev"
    if not os.path.exists(base):
        base = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev"
        
    train_path = os.path.join(base, "demos/Train.csv")
    test_path = os.path.join(base, "demos/TestInputSegments.csv")
    sample_sub_path = os.path.join(base, "demos/SampleSubmission.csv")
    
    print("Preparing ESN Data (Sequences)...")
    X_train, y_train, X_val, y_val = create_raw_sequences(train_path, val_split=0.2)
    print(f"Train Sequence Shape: {X_train.shape}, Val: {X_val.shape}")
    
    # ESN Params
    input_dim = X_train.shape[2] 
    esn = ESNClassifier(input_dim=input_dim, 
                        reservoir_dim=800, 
                        spectral_radius=0.95, 
                        leak_rate=0.2, 
                        sparsity=0.2,
                        alpha=10.0)
    
    print("Training ESN...")
    esn.fit(X_train, y_train)
    
    print("Evaluating...")
    val_pred = esn.predict(X_val)
    
    acc = accuracy_score(y_val, val_pred)
    f1 = f1_score(y_val, val_pred, average='macro')
    
    print(f"\nESN Validation Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Macro:  {f1:.4f}")
    
    # Inference
    print("\nStarting Inference...")
    congestion_map = {0: 'free flowing', 1: 'light delay', 2: 'moderate delay', 3: 'heavy delay'}
    test_df = pd.read_csv(test_path)
    prediction_dict = {}
    
    for label, group in test_df.groupby('view_label'):
        group = identify_blocks(group)
        for b_id, block in group.groupby('block_id'):
            # This logic needs to match create_raw_sequences utils but for inference
            feats, _ = get_features_and_labels(block)
            
            if len(feats) < 15:
                history = [feats[0]] * (15 - len(feats)) + list(feats)
            else:
                history = list(feats[-15:])
            
            start_id = int(round(history[-1][3] * 5000))
            
            for i in range(1, 9):
                # ESN needs (1, 15, F)
                current_seq = np.array(history[-15:]).reshape(1, 15, -1)
                
                p_idx = esn.predict(current_seq)[0]
                p_label = congestion_map[p_idx]
                prediction_dict[(label, start_id + i)] = p_label
                
                next_feat = np.copy(history[-1])
                curr_h = next_feat[0] * 23.0
                curr_m = next_feat[1] * 59.0
                curr_m += 5
                if curr_m > 59:
                    curr_m -= 60
                    curr_h = (curr_h + 1) % 24
                next_feat[0] = curr_h / 23.0
                next_feat[1] = curr_m / 59.0
                next_feat[3] += 1/5000.0
                history.append(next_feat)

    print("Mapping predictions...")
    sample_sub = pd.read_csv(sample_sub_path)
    final_targets = []
    
    for idx, row in sample_sub.iterrows():
        parts = row['ID'].split('_')
        tid = int(parts[2])
        vlabel = parts[3]
        final_targets.append(prediction_dict.get((vlabel, tid), 'free flowing'))
        
    sample_sub['Target'] = final_targets
    sample_sub['Target_Accuracy'] = final_targets
    
    output_dir = os.path.join(base, "submissions")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "submission_esn.csv")
    sample_sub.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")

if __name__ == "__main__":
    main()
