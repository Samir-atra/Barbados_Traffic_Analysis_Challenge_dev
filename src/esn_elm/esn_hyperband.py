"""Echo State Network (ESN) Hyperparameter Search.

This script performs a Randomized Search (Hyperband-style) for ESN hyperparameters
to maximize Validation F1-Macro score.

Search Space:
- Reservoir Dimension: 500 to 5000
- Spectral Radius: 0.1 to 1.5
- Leak Rate: 0.01 to 1.0
- Sparsity: 0.01 to 0.5
- Alpha (Readout Regularization): 1e-3 to 1e3 (Log scale)

This approach strictly avoids backpropagation.
"""

import os
import numpy as np
import pandas as pd
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from scipy import sparse

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
    
    # Enhanced feature set: 20 features total
    features = np.concatenate([
        df[['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'dow_sin', 'dow_cos']].values,
        df[['is_morning_rush', 'is_evening_rush']].values,
        df[['hour_norm', 'minute_norm', 'seg_id_norm']].values,
        df[['view_id']].values,
        view_1hot,
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
            
            n_block = len(block)
            n_train_rows = int(n_block * (1 - val_split))
            
            train_block = block.iloc[:n_train_rows]
            val_block = block.iloc[n_train_rows:]
            
            def make_seqs(sub_block):
                feats, labels = get_features_and_labels(sub_block)
                X_w, y_w = [], []
                for i in range(len(feats) - seq_len):
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
        self.reservoir_dim = reservoir_dim
        self.leak_rate = leak_rate
        self.alpha = alpha
        self.washout = washout
        self.state_avg_steps = state_avg_steps
        self.input_scale = input_scale
        self.rng = np.random.default_rng(seed)
        
        self.W_in = (self.rng.random((reservoir_dim, input_dim)) - 0.5) * input_scale
        
        W = sparse.random(reservoir_dim, reservoir_dim, density=sparsity, random_state=seed)
        eigenvalues = np.linalg.eigvals(W.toarray())
        max_eigen = np.max(np.abs(eigenvalues)) + 1e-10
        self.W_res = W * (spectral_radius / max_eigen)
        
        self.W_out = None
        self.scaler = StandardScaler()
        
    def _get_states(self, X):
        n_samples, n_steps, _ = X.shape
        states = np.zeros((n_samples, self.reservoir_dim))
        all_states = []
        
        for t in range(n_steps):
            u_t = X[:, t, :]
            input_part = np.dot(u_t, self.W_in.T)
            res_part = self.W_res.dot(states.T).T
            update = np.tanh(input_part + res_part)
            states = (1 - self.leak_rate) * states + self.leak_rate * update
            all_states.append(states.copy())
        
        # Multi-step state averaging
        if self.state_avg_steps > 1 and len(all_states) >= self.state_avg_steps:
            last_states = np.stack(all_states[-self.state_avg_steps:], axis=0)
            return np.mean(last_states, axis=0)
        else:
            return states
    
    def fit(self, X, y, class_weights=None):
        classes, counts = np.unique(y, return_counts=True)
        print(f"Input to Model - Class Counts: {dict(zip(classes, counts))}")
        N, T, F = X.shape
        X_flat = X.reshape(-1, F)
        self.scaler.fit(X_flat)
        X_scaled = self.scaler.transform(X_flat).reshape(N, T, F)
        
        states = self._get_states(X_scaled)
        states_bias = np.hstack([states, np.ones((N, 1))])
        
        Y_onehot = np.eye(4)[y]
        
        if class_weights is not None:
            sample_weights = np.array([class_weights[label] for label in y])
            S_sqrt = np.sqrt(sample_weights)[:, None]
            states_bias = states_bias * S_sqrt
            Y_onehot = Y_onehot * S_sqrt

        res_size = self.reservoir_dim + 1
        I = np.eye(res_size)
        A = np.dot(states_bias.T, states_bias) + self.alpha * I
        B = np.dot(states_bias.T, Y_onehot)
        
        self.W_out = np.linalg.solve(A, B)
        
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
    
    print("Preparing Data...")
    X_train, y_train, X_val, y_val = create_raw_sequences(train_path, val_split=0.2)
    
    t_cls, t_cnt = np.unique(y_train, return_counts=True)
    v_cls, v_cnt = np.unique(y_val, return_counts=True)
    print(f"Loaded Train class counts: {dict(zip(t_cls, t_cnt))}")
    print(f"Loaded Val class counts:   {dict(zip(v_cls, v_cnt))}")
    
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    
    # --- SEARCH CONFIG ---
    N_TRIALS = 50 # ESN is slower than ELM, starting with 50 trials
    input_dim = X_train.shape[2]
    
    search_space = {
        'reservoir_dim': lambda: np.random.randint(500, 5000),
        'spectral_radius': lambda: np.random.uniform(0.1, 1.5),
        'leak_rate': lambda: np.random.uniform(0.01, 1.0),
        'sparsity': lambda: np.random.uniform(0.01, 0.5),
        'alpha': lambda: 10 ** np.random.uniform(-3, 3),
        'input_scale': lambda: np.random.uniform(0.1, 3.0),
        'state_avg_steps': lambda: np.random.choice([1, 2, 3, 4, 5])
    }
    
    best_f1 = -1.0
    best_cfg = None
    best_model = None
    
    print(f"\nStarting Search ({N_TRIALS} trials)...")
    start_time = time.time()
    
    for i in range(N_TRIALS):
        cfg = {k: v() for k, v in search_space.items()}
        print(f"Trial {i+1}/{N_TRIALS}: {cfg}")
        
        try:
            esn = ESNClassifier(input_dim=input_dim, **cfg, seed=42)
            esn.fit(X_train, y_train, class_weights=class_weight_dict)
            
            # Evaluate on Train
            train_pred = esn.predict(X_train)
            train_f1 = f1_score(y_train, train_pred, average='macro')
            
            # Evaluate on Val
            val_pred = esn.predict(X_val)
            val_f1 = f1_score(y_val, val_pred, average='macro')
            
            print(f"  -> Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_cfg = cfg
                best_model = esn
                print(f"  >>> NEW BEST! Val F1: {best_f1:.4f}")
                
        except Exception as e:
            print(f"  - Failed: {e}")
            
    print(f"\nSearch Complete in {time.time() - start_time:.1f}s")
    print(f"Best Configuration: {best_cfg}")
    print(f"Best Validation F1: {best_f1:.4f}")
    
    # --- INFERENCE ---
    if best_model is not None:
        print(f"\nGenerating Submission with Best Model...")
        congestion_map = {0: 'free flowing', 1: 'light delay', 2: 'moderate delay', 3: 'heavy delay'}
        test_df = pd.read_csv(test_path)
        prediction_dict = {}
        
        for label, group in test_df.groupby('view_label'):
            group = identify_blocks(group)
            for b_id, block in group.groupby('block_id'):
                feats, _ = get_features_and_labels(block)
                if len(feats) < 15:
                    history = [feats[0]] * (15 - len(feats)) + list(feats)
                else:
                    history = list(feats[-15:])
                
                start_id = int(round(history[-1][3] * 5000))
                for j in range(1, 9):
                    current_seq = np.array(history[-15:]).reshape(1, 15, -1)
                    p_idx = best_model.predict(current_seq)[0]
                    prediction_dict[(label, start_id + j)] = congestion_map[p_idx]
                    
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

        sample_sub = pd.read_csv(sample_sub_path)
        final_targets = []
        for idx, row in sample_sub.iterrows():
            parts = row['ID'].split('_')
            final_targets.append(prediction_dict.get((parts[3], int(parts[2])), 'free flowing'))
            
        sample_sub['Target'] = final_targets
        sample_sub['Target_Accuracy'] = final_targets
        
        output_dir = os.path.join(base, "submissions")
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, "submission_esn_hyperband.csv")
        sample_sub.to_csv(out_file, index=False)
        print(f"Saved: {out_file}")

if __name__ == "__main__":
    main()
