"""Extreme Learning Machine (ELM) with Grid Search and Ensembling.

This script implements an advanced ELM approach for traffic prediction, including:
1. Hyperparameter Grid Search (Hidden Units, Alpha, Activation, Scale).
2. Ensemble Voting (Training multiple diverse ELMs).
3. Weighted Ridge Regression to handle class imbalance.

This approach STRICTLY avoids backpropagation.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

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
    """Processes CSV into sliding window samples (flattened)."""
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

    train_y_np, val_y_np = np.array(train_y), np.array(val_y)
    t_cls, t_cnt = np.unique(train_y_np, return_counts=True)
    v_cls, v_cnt = np.unique(val_y_np, return_counts=True)
    print(f"Loaded {len(train_y_np)} train samples with class counts: {dict(zip(t_cls, t_cnt))}")
    print(f"Loaded {len(val_y_np)} val samples with class counts: {dict(zip(v_cls, v_cnt))}")
                
    return np.array(train_X), train_y_np, np.array(val_X), val_y_np

class ELMClassifier:
    """Extreme Learning Machine for Multi-class Classification."""
    
    def __init__(self, hidden_units=1000, alpha=1.0, activation='relu', scale=1.0, seed=42):
        """
        Args:
            hidden_units: Number of random hidden neurons.
            alpha: L2 regularization strength.
            activation: 'tanh', 'sigmoid', 'relu', or 'sine'.
            scale: Scaling factor for input weights (controls saturation).
        """
        self.hidden_units = hidden_units
        self.alpha = alpha
        self.activation = activation
        self.scale = scale
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
        elif self.activation == 'sine':
            return np.sin(H)
        return H

    def fit(self, X, y, class_weights=None):
        classes, counts = np.unique(y, return_counts=True)
        print(f"Input to Model - Class Counts: {dict(zip(classes, counts))}")
        """Train the ELM with optional class weighting."""
        # 1. Scale Input
        X = self.scaler.fit_transform(X)
        n_samples, n_features = X.shape
        
        # 2. Initialize Random Weights (Scaled)
        # Using uniform/normal based on preference. Normal is standard.
        self.input_weights = self.rng.normal(scale=self.scale, size=(n_features, self.hidden_units))
        self.bias = self.rng.normal(scale=self.scale, size=(self.hidden_units,))
        
        # 3. Compute Hidden Layer Output Matrix H
        H = np.dot(X, self.input_weights) + self.bias
        H = self._activate(H) # (N, hidden_units)
        
        # 4. Preparing Targets (One-Hot)
        n_classes = 4
        Y_onehot = np.zeros((n_samples, n_classes))
        Y_onehot[np.arange(n_samples), y] = 1
        
        # Apply Class Weights if provided
        if class_weights is not None:
            # Create sample weight vector
            sample_weights = np.array([class_weights[label] for label in y])
            W_sqrt = np.sqrt(sample_weights)[:, None] # (N, 1)
            H = H * W_sqrt
            Y_onehot = Y_onehot * W_sqrt

        # 5. Solve for Output Weights (Ridge Regression)
        I = np.eye(self.hidden_units)
        
        A = np.dot(H.T, H) + self.alpha * I
        B = np.dot(H.T, Y_onehot)
        
        print(f"  - Solving ({self.hidden_units} units, act={self.activation}, alpha={self.alpha})...")
        try:
            self.output_weights = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            self.output_weights = np.dot(np.linalg.pinv(A), B)
            
    def predict_proba(self, X):
        X = self.scaler.transform(X)
        H = np.dot(X, self.input_weights) + self.bias
        H = self._activate(H)
        return np.dot(H, self.output_weights)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

def main():
    base = "/teamspace/studios/this_studio/Barbados_Traffic_Analysis_Challenge_dev"
    if not os.path.exists(base):
        base = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev"
        
    train_path = os.path.join(base, "demos/Train.csv")
    test_path = os.path.join(base, "demos/TestInputSegments.csv")
    sample_sub_path = os.path.join(base, "demos/SampleSubmission.csv")
    
    print("Preparing ELM Data...")
    X_train, y_train, X_val, y_val = create_dataset_splits(train_path, val_split=0.2)
    
    t_cls, t_cnt = np.unique(y_train, return_counts=True)
    v_cls, v_cnt = np.unique(y_val, return_counts=True)
    print(f"Loaded Train class counts: {dict(zip(t_cls, t_cnt))}")
    print(f"Loaded Val class counts:   {dict(zip(v_cls, v_cnt))}")
    
    print(f"Train Shape: {X_train.shape}, Val Shape: {X_val.shape}")
    
    # Calculate Class Weights
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    print(f"Class Weights: {class_weight_dict}")
    
    # --- Ensemble Training & Evaluation ---
    print("\nTraining & Evaluating Individual ELMs...")
    
    # Grid of diversity (Manually selected diverse configs)
    configs = [
        {'hidden': 3000, 'alpha': 100.0, 'act': 'relu', 'scale': 1.0},
        {'hidden': 3000, 'alpha': 500.0, 'act': 'relu', 'scale': 1.0},
        {'hidden': 3000, 'alpha': 10.0,  'act': 'tanh', 'scale': 0.5},
        {'hidden': 3000, 'alpha': 50.0,  'act': 'tanh', 'scale': 1.0},
        {'hidden': 3000, 'alpha': 50.0,  'act': 'leaky_relu', 'scale': 1.0},
        {'hidden': 3000, 'alpha': 100.0, 'act': 'leaky_relu', 'scale': 1.0},
    ]
    
    models = []
    best_f1 = -1.0
    best_cfg = None
    
    val_probs_sum = np.zeros((len(X_val), 4))
    
    for idx, cfg in enumerate(configs):
        print(f"\nModel {idx+1}/{len(configs)}: {cfg}")
        elm = ELMClassifier(hidden_units=cfg['hidden'], 
                           alpha=cfg['alpha'], 
                           activation=cfg['act'], 
                           scale=cfg['scale'],
                           seed=42 + idx)
        elm.fit(X_train, y_train, class_weights=class_weight_dict)
        
        # Evaluate Individual Model
        val_probs = elm.predict_proba(X_val)
        val_pred = np.argmax(val_probs, axis=1)
        
        f1 = f1_score(y_val, val_pred, average='macro')
        acc = accuracy_score(y_val, val_pred)
        
        print(f"  -> Accuracy: {acc:.4f}, F1 Macro: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_cfg = cfg
            
        models.append(elm)
        val_probs_sum += val_probs

    print(f"\n>>> Best Single Model Config: {best_cfg}")
    print(f">>> Best Single F1 Score: {best_f1:.4f}")
        
    # --- Ensemble Evaluation ---
    print("\nEvaluating Ensemble...")
    val_pred_ensemble = np.argmax(val_probs_sum, axis=1)
    
    acc = accuracy_score(y_val, val_pred_ensemble)
    f1 = f1_score(y_val, val_pred_ensemble, average='macro')
    prec = precision_score(y_val, val_pred_ensemble, average='macro')
    rec = recall_score(y_val, val_pred_ensemble, average='macro')
    
    print(f"\n>>> Ensemble Validation Metrics <<<")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Macro:  {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    
    # --- Inference with Best Model ---
    print(f"\nStarting Inference with Best Model ({best_cfg})...")
    
    # Re-train best model on FULL training data (optional but recommended)
    # For now, we reuse the trained instance corresponding to best_cfg
    # We need to find the model instance in the list that matches best_cfg
    best_model_idx = configs.index(best_cfg)
    best_model = models[best_model_idx]
    
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
                
                # Single Best Model Prediction
                p_idx = best_model.predict(current_window)[0]
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
    out_file = os.path.join(output_dir, "submission_elm_ensemble.csv")
    sample_sub.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")

if __name__ == "__main__":
    main()
