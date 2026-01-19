"""Extreme Learning Machine (ELM) for Traffic Prediction.

This script implements an Extreme Learning Machine (ELM) to predict traffic
congestion. ELMs are Single-Hidden Layer Feedforward Networks (SLFNs) where
input weights/biases are random and fixed, and output weights are analytically 
computed using the Moore-Penrose pseudoinverse or Ridge Regression.

This approach STRICTLY avoids backpropagation.

Workflow:
1. Load Train.csv and format into sliding windows (flattened).
2. Train ELM using ridge regression on the hidden layer representations.
3. Evaluate on validation split.
4. Generate submission for TestInputSegments.csv.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import joblib

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)

def get_features_and_labels(df):
    """Extracts features and labels (same as FF script)."""
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
    
    # signaling feature mapping
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
    ], axis=1).astype('float32')

    return features, labels

def identify_blocks(group):
    """Identify continuous sequential blocks."""
    group = group.sort_values('time_segment_id')
    ids = group['time_segment_id'].values
    is_break = np.zeros(len(ids), dtype=int)
    is_break[1:] = (ids[1:] != ids[:-1] + 1).astype(int)
    group['block_id'] = np.cumsum(is_break)
    return group

def create_dataset_splits(csv_path, val_split=0.2, seq_len=15):
    """Processes CSV into sliding window samples (flattened)."""
    df = pd.read_csv(csv_path)
    train_X, train_y = [], []
    val_X, val_y = [], []
    
    print(f"Loading {len(df)} rows from {csv_path}...")
    for label, group in df.groupby('view_label'):
        group = identify_blocks(group)
        
        for b_id, block in group.groupby('block_id'):
            if len(block) < seq_len + 1: continue
            
            # Split block first to avoid leakage
            n_block = len(block)
            n_train_rows = int(n_block * (1 - val_split))
            
            train_block = block.iloc[:n_train_rows]
            val_block = block.iloc[n_train_rows:]
            
            # Function to create windows
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
    """Extreme Learning Machine for Multi-class Classification."""
    
    def __init__(self, hidden_units=1000, alpha=1.0, activation='tanh', seed=42):
        """
        Args:
            hidden_units: Number of random hidden neurons.
            alpha: L2 regularization strength (Ridge regression parameter).
            activation: 'tanh', 'sigmoid', 'relu'.
        """
        self.hidden_units = hidden_units
        self.alpha = alpha
        self.activation = activation
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        self.input_weights = None
        self.bias = None
        self.output_weights = None
        self.scaler = StandardScaler()
        
    def _activate(self, H):
        if self.activation == 'tanh':
            return np.tanh(H)
        elif self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-H))
        elif self.activation == 'relu':
            return np.maximum(0, H)
        return H

    def fit(self, X, y):
        """Train the ELM.
        
        Args:
            X: Input features (N_samples, n_features)
            y: Integer labels (N_samples,)
        """
        # 1. Scale Input
        X = self.scaler.fit_transform(X)
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # 2. Initialize Random Weights (Fixed)
        self.input_weights = self.rng.normal(size=(n_features, self.hidden_units))
        self.bias = self.rng.normal(size=(self.hidden_units,))
        
        # 3. Compute Hidden Layer Output Matrix H
        H = np.dot(X, self.input_weights) + self.bias
        H = self._activate(H) # (N, hidden_units)
        
        # 4. Preparing Targets (One-Hot)
        Y_onehot = np.zeros((n_samples, 4)) # Assuming 4 classes 0-3
        Y_onehot[np.arange(n_samples), y] = 1
        
        # 5. Solve for Output Weights (Ridge Regression)
        # W_out = (H^T H + alpha * I)^-1 H^T Y
        # Using linear solver is more stable than explicit inverse
        
        I = np.eye(self.hidden_units)
        A = np.dot(H.T, H) + self.alpha * I
        B = np.dot(H.T, Y_onehot)
        
        print(f"Solving linear system for {self.hidden_units} hidden units...")
        try:
            self.output_weights = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            print("Matrix singular, using pseudoinverse...")
            self.output_weights = np.dot(np.linalg.pinv(A), B)
            
        print("ELM Training Complete.")
        
    def predict(self, X):
        X = self.scaler.transform(X)
        H = np.dot(X, self.input_weights) + self.bias
        H = self._activate(H)
        Y_pred = np.dot(H, self.output_weights)
        return np.argmax(Y_pred, axis=1)

def main():
    base = "/teamspace/studios/this_studio/Barbados_Traffic_Analysis_Challenge_dev"
    # Adjust base path for local environment if needed
    if not os.path.exists(base):
        base = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev"
        
    train_path = os.path.join(base, "demos/Train.csv")
    test_path = os.path.join(base, "demos/TestInputSegments.csv")
    sample_sub_path = os.path.join(base, "demos/SampleSubmission.csv")
    
    print("Preparing ELM Data...")
    X_train, y_train, X_val, y_val = create_dataset_splits(train_path, val_split=0.2)
    print(f"Train Shape: {X_train.shape}, Val Shape: {X_val.shape}")
    
    # ELM Hyperparameters
    # High number of hidden units is key for ELM
    elm = ELMClassifier(hidden_units=20000, alpha=10.0, activation='relu', seed=42)
    
    print("Training ELM...")
    elm.fit(X_train, y_train)
    
    print("Evaluating...")
    val_pred = elm.predict(X_val)
    
    acc = accuracy_score(y_val, val_pred)
    f1 = f1_score(y_val, val_pred, average='macro')
    prec = precision_score(y_val, val_pred, average='macro')
    rec = recall_score(y_val, val_pred, average='macro')
    
    print(f"\nELM Validation Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Macro:  {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    
    # --- Inference ---
    print("\nStarting Inference...")
    congestion_map = {0: 'free flowing', 1: 'light delay', 2: 'moderate delay', 3: 'heavy delay'}
    test_df = pd.read_csv(test_path)
    prediction_dict = {}
    
    # IMPORTANT: Need to recreate scaler fit on full training data ideally, 
    # but reuse the one from training split for consistency here.
    
    for label, group in test_df.groupby('view_label'):
        group = identify_blocks(group)
        for b_id, block in group.groupby('block_id'):
            feats, _ = get_features_and_labels(block)
            
            # Context buffer
            if len(feats) < 15:
                history = [feats[0]] * (15 - len(feats)) + list(feats)
            else:
                history = list(feats[-15:])
            
            # Start predicting 8 steps
            start_id = int(round(history[-1][3] * 5000))
            
            for i in range(1, 9):
                current_window = np.array(history[-15:]).flatten().reshape(1, -1)
                
                # Predict
                p_idx = elm.predict(current_window)[0]
                p_label = congestion_map[p_idx]
                
                target_id = start_id + i
                prediction_dict[(label, target_id)] = p_label
                
                # Update history (autoregressive)
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
    sample_sub['Target_Accuracy'] = final_targets # Match format
    
    output_dir = os.path.join(base, "submissions")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "submission_elm.csv")
    sample_sub.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")

if __name__ == "__main__":
    main()
