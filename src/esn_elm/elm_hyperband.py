"""ELM Hyperparameter Search using Random Search (Hyperband-style wide exploration).

Since ELM training is instantaneous (no epochs), we perform a massive Randomized Search
over a wide range of hyperparameters to maximize Validation F1-Macro score.

Search Space:
- Hidden Units: 500 to 10000
- Alpha (Regularization): 1e-3 to 1e3 (Log scale)
- Activation: ['relu', 'tanh', 'sigmoid', 'sine', 'leaky_relu', 'hard_sigmoid']
- Input Scaling: 0.1 to 10.0
"""

import os
import numpy as np
import pandas as pd
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Set seeds for reproducibility of the search splits
SEED = 42
np.random.seed(SEED)

def get_features_and_labels(df):
    """Extracts features and labels."""
    df = df.copy()
    df['video_time'] = pd.to_datetime(df['video_time'])
    df['hour'] = df['video_time'].dt.hour / 23.0
    df['minute'] = df['video_time'].dt.minute / 59.0
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek / 6.0
    
    view_map = {'Norman Niles #1': 0, 'Norman Niles #2': 1, 'Norman Niles #3': 2, 'Norman Niles #4': 3}
    df['view_id'] = df['view_label'].map(view_map)
    df['seg_id_norm'] = df['time_segment_id'] / 5000.0
    
    congestion_map = {'free flowing': 0, 'light delay': 1, 'moderate delay': 2, 'heavy delay': 3}
    if 'congestion_enter_rating' in df.columns:
        df['enter_id'] = df['congestion_enter_rating'].map(congestion_map).fillna(0).astype(int)
        labels = df['enter_id'].values
    else:
        labels = None
    
    view_1hot = pd.get_dummies(df['view_id'], prefix='view').reindex(columns=['view_0', 'view_1', 'view_2', 'view_3'], fill_value=0).astype(float).values
    
    if 'signaling' in df.columns:
        sig_map = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
        df['sig_id'] = df['signaling'].map(sig_map).fillna(0)
    else:
        df['sig_id'] = 0
        
    sig_1hot = pd.get_dummies(df['sig_id'], prefix='sig').reindex(columns=['sig_0', 'sig_1', 'sig_2', 'sig_3'], fill_value=0).astype(float).values
    
    features = np.concatenate([
        df[['hour', 'minute', 'day_of_week', 'seg_id_norm', 'view_id']].values,
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

def create_dataset_splits(csv_path, val_split=0.2, seq_len=15):
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
            
            def make_windows(sub_block):
                feats, labels = get_features_and_labels(sub_block)
                X_w, y_w = [], []
                for i in range(len(feats) - seq_len):
                    X_w.append(feats[i : i + seq_len].flatten())
                    y_w.append(labels[i + seq_len])
                return X_w, y_w
            
            if len(train_block) >= seq_len + 1:
                tx, ty = make_windows(train_block)
                train_X.extend(tx)
                train_y.extend(ty)
                
            if len(val_block) >= seq_len + 1:
                vx, vy = make_windows(val_block)
                val_X.extend(vx)
                val_y.extend(vy)
                
    return np.array(train_X), np.array(train_y), np.array(val_X), np.array(val_y)

class ELMClassifier:
    def __init__(self, hidden_units=1000, alpha=1.0, activation='relu', scale=1.0, seed=42):
        self.hidden_units = hidden_units
        self.alpha = alpha
        self.activation = activation
        self.scale = scale
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.scaler = StandardScaler()
        
    def _activate(self, H):
        if self.activation == 'tanh': return np.tanh(H)
        elif self.activation == 'sigmoid': return 1.0 / (1.0 + np.exp(-H))
        elif self.activation == 'relu': return np.maximum(0, H)
        elif self.activation == 'leaky_relu': return np.maximum(0.1 * H, H)
        elif self.activation == 'sine': return np.sin(H)
        elif self.activation == 'hard_sigmoid': return np.clip(0.2 * H + 0.5, 0.0, 1.0)
        return H

    def fit(self, X, y, class_weights=None):
        X = self.scaler.fit_transform(X)
        n_samples, n_features = X.shape
        
        self.input_weights = self.rng.normal(scale=self.scale, size=(n_features, self.hidden_units))
        self.bias = self.rng.normal(scale=self.scale, size=(self.hidden_units,))
        
        H = self._activate(np.dot(X, self.input_weights) + self.bias)
        
        Y_onehot = np.zeros((n_samples, 4))
        Y_onehot[np.arange(n_samples), y] = 1
        
        if class_weights is not None:
            sample_weights = np.array([class_weights[label] for label in y])
            W_sqrt = np.sqrt(sample_weights)[:, None]
            H = H * W_sqrt
            Y_onehot = Y_onehot * W_sqrt

        I = np.eye(self.hidden_units)
        A = np.dot(H.T, H) + self.alpha * I
        B = np.dot(H.T, Y_onehot)
        
        try:
            self.output_weights = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            self.output_weights = np.dot(np.linalg.pinv(A), B)
            
    def predict(self, X):
        X = self.scaler.transform(X)
        H = self._activate(np.dot(X, self.input_weights) + self.bias)
        return np.argmax(np.dot(H, self.output_weights), axis=1)

def main():
    base = "/teamspace/studios/this_studio/Barbados_Traffic_Analysis_Challenge_dev"
    if not os.path.exists(base):
        base = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev"
        
    train_path = os.path.join(base, "demos/Train.csv")
    test_path = os.path.join(base, "demos/TestInputSegments.csv")
    sample_sub_path = os.path.join(base, "demos/SampleSubmission.csv")
    
    print("Preparing Data...")
    X_train, y_train, X_val, y_val = create_dataset_splits(train_path, val_split=0.2)
    
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    
    # --- SEARCH CONFIG ---
    N_TRIALS = 100
    
    search_space = {
        'hidden_units': lambda: np.random.randint(500, 10000),
        'alpha': lambda: 10 ** np.random.uniform(-2, 3), # 0.01 to 1000
        'activation': lambda: np.random.choice(['relu', 'tanh', 'sigmoid', 'sine', 'leaky_relu']),
        'scale': lambda: np.random.uniform(0.1, 3.0)
    }
    
    best_f1 = -1.0
    best_cfg = None
    best_model = None
    
    print(f"\nStarting Search ({N_TRIALS} trials)...")
    start_time = time.time()
    
    for i in range(N_TRIALS):
        # Sample configuration
        cfg = {k: v() for k, v in search_space.items()}
        
        print(f"Trial {i+1}/{N_TRIALS}: {cfg}", end=" ... ")
        
        try:
            elm = ELMClassifier(**cfg, seed=42) # Keep seed fixed for reproducibility of weights given config
            elm.fit(X_train, y_train, class_weights=class_weight_dict)
            
            val_pred = elm.predict(X_val)
            f1 = f1_score(y_val, val_pred, average='macro')
            
            print(f"F1: {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_cfg = cfg
                best_model = elm
                print(f"  >>> NEW BEST! F1: {best_f1:.4f}")
                
        except Exception as e:
            print(f"Failed: {e}")
            
    print(f"\nSearch Complete in {time.time() - start_time:.1f}s")
    print(f"Best Configuration: {best_cfg}")
    print(f"Best Validation F1: {best_f1:.4f}")
    
    # --- INFERENCE ---
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
            
            for i in range(1, 9):
                current_window = np.array(history[-15:]).flatten().reshape(1, -1)
                p_idx = best_model.predict(current_window)[0]
                prediction_dict[(label, start_id + i)] = congestion_map[p_idx]
                
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
    
    out_file = os.path.join(base, "submissions/submission_elm_hyperband.csv")
    sample_sub.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")

if __name__ == "__main__":
    main()
