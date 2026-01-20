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
1. Load Train.csv with improved chunked data loading.
2. Run sequences through ESN Reservoir.
3. Train Readout Layer (Ridge Regression).
4. Predict on Test Set.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from scipy import sparse

# Add data_processing to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
from chunked_data_loader import create_raw_sequences_chunked, CHUNK_SIZE, SEQ_LEN

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)


def get_features_and_labels(df):
    """Extracts features and labels with enhanced feature engineering."""
    df = df.copy()
    df['video_time'] = pd.to_datetime(df['video_time'])
    
    # Extract raw time values
    hour = df['video_time'].dt.hour
    minute = df['video_time'].dt.minute
    day_of_week = pd.to_datetime(df['date']).dt.dayofweek
    
    # Cyclical encoding for hour (captures 24-hour cycle)
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    
    # Cyclical encoding for minute
    df['minute_sin'] = np.sin(2 * np.pi * minute / 60.0)
    df['minute_cos'] = np.cos(2 * np.pi * minute / 60.0)
    
    # Cyclical encoding for day of week
    df['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7.0)
    
    # Peak hour indicators (rush hours: 7-9 AM, 4-6 PM)
    df['is_morning_rush'] = ((hour >= 7) & (hour <= 9)).astype(float)
    df['is_evening_rush'] = ((hour >= 16) & (hour <= 18)).astype(float)
    
    # Linear time features (keep for additional signal)
    df['hour_norm'] = hour / 23.0
    df['minute_norm'] = minute / 59.0
    
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
    
    # Enhanced feature set: 21 features total
    features = np.concatenate([
        # Cyclical time features (6)
        df[['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'dow_sin', 'dow_cos']].values,
        # Peak hour indicators (2)
        df[['is_morning_rush', 'is_evening_rush']].values,
        # Linear time + segment (3)
        df[['hour_norm', 'minute_norm', 'seg_id_norm']].values,
        # View ID (1) + one-hot (4)
        df[['view_id']].values,
        view_1hot,
        # Signaling one-hot (4)
        sig_1hot
    ], axis=1).astype('float32')

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

    train_y_np, val_y_np = np.array(train_y), np.array(val_y)
    t_cls, t_cnt = np.unique(train_y_np, return_counts=True)
    v_cls, v_cnt = np.unique(val_y_np, return_counts=True)
    print(f"Loaded {len(train_y_np)} train samples with class counts: {dict(zip(t_cls, t_cnt))}")
    print(f"Loaded {len(val_y_np)} val samples with class counts: {dict(zip(v_cls, v_cnt))}")
                
    return np.array(train_X), train_y_np, np.array(val_X), val_y_np

class ESNClassifier:
    """Echo State Network for Sequence Classification."""
    
    def __init__(self, input_dim, reservoir_dim=500, spectral_radius=0.9, 
                 leak_rate=0.3, sparsity=0.1, alpha=1.0, washout=0, 
                 state_avg_steps=3, input_scale=1.0, seed=42):
        """
        Args:
            input_dim: Number of input features.
            reservoir_dim: Neurons in reservoir.
            spectral_radius: Scaling of reservoir weight matrix (rho).
            leak_rate: Leaking rate (alpha) in state update.
            sparsity: Probability of connection in reservoir.
            alpha: Ridge regression regularization.
            washout: Number of initial time steps to discard (warm-up period).
            state_avg_steps: Number of final states to average for classification.
            input_scale: Scaling factor for input weights.
        """
        self.reservoir_dim = reservoir_dim
        self.leak_rate = leak_rate
        self.alpha = alpha
        self.washout = washout
        self.state_avg_steps = state_avg_steps
        self.input_scale = input_scale
        self.rng = np.random.default_rng(seed)
        
        # 1. Input Weights (Dense, Random, Scaled)
        self.W_in = (self.rng.random((reservoir_dim, input_dim)) - 0.5) * input_scale
        
        # 2. Reservoir Weights (Sparse, Random)
        W = sparse.random(reservoir_dim, reservoir_dim, density=sparsity, random_state=seed)
        eigenvalues = np.linalg.eigvals(W.toarray())
        max_eigen = np.max(np.abs(eigenvalues)) + 1e-10
        self.W_res = W * (spectral_radius / max_eigen)
        
        self.W_out = None
        self.scaler = StandardScaler()
        
    def _get_states(self, X):
        """Run reservoir dynamics on batch of sequences."""
        n_samples, n_steps, _ = X.shape
        
        states = np.zeros((n_samples, self.reservoir_dim))
        all_states = []  # Store all states for multi-step averaging
        
        for t in range(n_steps):
            u_t = X[:, t, :]
            
            input_part = np.dot(u_t, self.W_in.T)
            res_part = self.W_res.dot(states.T).T
            
            update = np.tanh(input_part + res_part)
            states = (1 - self.leak_rate) * states + self.leak_rate * update
            
            all_states.append(states.copy())
        
        # Multi-step state averaging: average the last N states
        if self.state_avg_steps > 1 and len(all_states) >= self.state_avg_steps:
            # Stack last N states and average
            last_states = np.stack(all_states[-self.state_avg_steps:], axis=0)  # (N_steps, N_samples, Res)
            avg_states = np.mean(last_states, axis=0)  # (N_samples, Res)
            return avg_states
        else:
            return states  # Return final state only
    
    def fit(self, X, y, class_weights=None):
        classes, counts = np.unique(y, return_counts=True)
        print(f"Input to Model - Class Counts: {dict(zip(classes, counts))}")
        
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
        
        # Apply Class Weights if provided
        if class_weights is not None:
            sample_weights = np.array([class_weights[label] for label in y])
            S_sqrt = np.sqrt(sample_weights)[:, None]
            states_bias = states_bias * S_sqrt
            Y_onehot = Y_onehot * S_sqrt
        
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
    
    print("Preparing ESN Data (Chunked Sequences)...")
    X_train, y_train, X_val, y_val = create_raw_sequences_chunked(
        train_path, val_split=0.2, seq_len=SEQ_LEN, chunk_size=CHUNK_SIZE)
    
    t_cls, t_cnt = np.unique(y_train, return_counts=True)
    v_cls, v_cnt = np.unique(y_val, return_counts=True)
    print(f"Train data classes: {dict(zip(t_cls, t_cnt))}")
    print(f"Val data classes:   {dict(zip(v_cls, v_cnt))}")
    print(f"Train Sequence Shape: {X_train.shape}, Val: {X_val.shape}")
    
    # ESN Params - Optimized for performance
    input_dim = X_train.shape[2] 
    esn = ESNClassifier(input_dim=input_dim, 
                        reservoir_dim=2500,  # Balanced size
                        spectral_radius=0.95, 
                        leak_rate=0.2, 
                        sparsity=0.05,
                        alpha=50.0,
                        input_scale=0.8,
                        state_avg_steps=5,
                        seed=42)
    
    print("Training ESN...")
    # Calculate Class Weights to handle imbalance
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    print(f"Class Weights: {class_weight_dict}")
    
    esn.fit(X_train, y_train, class_weights=class_weight_dict)
    
    print("\nEvaluating on Training Data...")
    train_pred = esn.predict(X_train)
    train_f1 = f1_score(y_train, train_pred, average='macro')
    train_acc = accuracy_score(y_train, train_pred)
    print(f"Train Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")

    print("\nEvaluating on Validation Data...")
    val_pred = esn.predict(X_val)
    
    acc = accuracy_score(y_val, val_pred)
    f1 = f1_score(y_val, val_pred, average='macro')
    prec = precision_score(y_val, val_pred, average='macro')
    rec = recall_score(y_val, val_pred, average='macro')
    
    print(f"Val Accuracy:  {acc:.4f}")
    print(f"Val F1 Macro:  {f1:.4f}")
    print(f"Val Precision: {prec:.4f}")
    print(f"Val Recall:    {rec:.4f}")
    
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
