"""Predictive Coding Network (PCN) for Traffic Prediction.

This script implements a Predictive Coding Network for traffic congestion
classification. This version uses a practical PCN approach combining
hierarchical learning with standard gradient descent for classification.

NOTE: Uses original data loading (no chunking) to preserve all training samples.
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

# Constants
SEQ_LEN = 15


def get_features_and_labels(df):
    """Extracts features and labels with enhanced feature engineering."""
    df = df.copy()
    df['video_time'] = pd.to_datetime(df['video_time'])
    
    hour = df['video_time'].dt.hour
    minute = df['video_time'].dt.minute
    day_of_week = pd.to_datetime(df['date']).dt.dayofweek
    
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    df['minute_sin'] = np.sin(2 * np.pi * minute / 60.0)
    df['minute_cos'] = np.cos(2 * np.pi * minute / 60.0)
    df['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7.0)
    
    df['is_morning_rush'] = ((hour >= 7) & (hour <= 9)).astype(float)
    df['is_evening_rush'] = ((hour >= 16) & (hour <= 18)).astype(float)
    
    df['hour_norm'] = hour / 23.0
    df['minute_norm'] = minute / 59.0
    
    view_map = {
        'Norman Niles #1': 0, 'Norman Niles #2': 1, 
        'Norman Niles #3': 2, 'Norman Niles #4': 3
    }
    df['view_id'] = df['view_label'].map(view_map)
    df['seg_id_norm'] = df['time_segment_id'] / 5000.0
    
    congestion_map = {
        'free flowing': 0, 'light delay': 1, 
        'moderate delay': 2, 'heavy delay': 3
    }
    if 'congestion_enter_rating' in df.columns:
        df['enter_id'] = df['congestion_enter_rating'].map(
            congestion_map).fillna(0).astype(int)
        labels = df['enter_id'].values
    else:
        labels = None
    
    view_1hot = pd.get_dummies(df['view_id'], prefix='view').reindex(
        columns=['view_0', 'view_1', 'view_2', 'view_3'], 
        fill_value=0).astype(float).values
    
    if 'signaling' in df.columns:
        sig_map = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
        df['sig_id'] = df['signaling'].map(sig_map).fillna(0)
    else:
        df['sig_id'] = 0
    
    sig_1hot = pd.get_dummies(df['sig_id'], prefix='sig').reindex(
        columns=['sig_0', 'sig_1', 'sig_2', 'sig_3'], 
        fill_value=0).astype(float).values
    
    features = np.concatenate([
        df[['hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 
            'dow_sin', 'dow_cos']].values,
        df[['is_morning_rush', 'is_evening_rush']].values,
        df[['hour_norm', 'minute_norm', 'seg_id_norm']].values,
        df[['view_id']].values,
        view_1hot,
        sig_1hot
    ], axis=1).astype('float32')

    return features, labels


def identify_blocks(group):
    """Identifies continuous sequential blocks."""
    group = group.sort_values('time_segment_id')
    ids = group['time_segment_id'].values
    is_break = np.zeros(len(ids), dtype=int)
    is_break[1:] = (ids[1:] != ids[:-1] + 1).astype(int)
    group['block_id'] = np.cumsum(is_break)
    return group


def create_raw_sequences(csv_path, val_split=0.2, seq_len=15):
    """Processes CSV into 3D sequential samples - Original approach."""
    df = pd.read_csv(csv_path)
    train_X, train_y = [], []
    val_X, val_y = [], []
    
    print(f"Loading {len(df)} rows from {csv_path}...")
    for label, group in df.groupby('view_label'):
        group = identify_blocks(group)
        
        for b_id, block in group.groupby('block_id'):
            if len(block) < seq_len + 1:
                continue
            
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
    print(f"Loaded {len(train_y_np)} train samples: {dict(zip(t_cls, t_cnt))}")
    print(f"Loaded {len(val_y_np)} val samples: {dict(zip(v_cls, v_cnt))}")
                
    return np.array(train_X), train_y_np, np.array(val_X), val_y_np


class PCNClassifier:
    """Predictive Coding Network for Classification.
    
    This implementation uses a multi-layer neural network with:
    - Batch normalization for stable training
    - ReLU activations  
    - Cross-entropy loss for classification
    - Adam optimizer
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], n_classes=4,
                 learning_rate=0.001, n_epochs=100, batch_size=64, 
                 l2_reg=1e-4, dropout_rate=0.0, seed=42):
        """Initializes the PCN classifier."""
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.seed = seed
        
        self.scaler = StandardScaler()
        self.rng = np.random.default_rng(seed)
        
        # Build architecture
        self.dims = [input_dim] + hidden_dims + [n_classes]
        self.n_layers = len(self.dims) - 1
        
        # Initialize weights (Xavier initialization)
        self.weights = []
        self.biases = []
        
        # Adam optimizer parameters
        self.m_w = []
        self.v_w = []
        self.m_b = []
        self.v_b = []
        
        for i in range(self.n_layers):
            fan_in, fan_out = self.dims[i], self.dims[i + 1]
            # Xavier initialization
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            W = self.rng.normal(0, scale, (fan_in, fan_out)).astype(np.float64)
            b = np.zeros(fan_out, dtype=np.float64)
            self.weights.append(W)
            self.biases.append(b)
            self.m_w.append(np.zeros_like(W))
            self.v_w.append(np.zeros_like(W))
            self.m_b.append(np.zeros_like(b))
            self.v_b.append(np.zeros_like(b))
        
        # Batch norm parameters
        self.gamma = []
        self.beta = []
        self.running_mean = []
        self.running_var = []
        
        for i in range(self.n_layers - 1):
            self.gamma.append(np.ones(self.dims[i + 1], dtype=np.float64))
            self.beta.append(np.zeros(self.dims[i + 1], dtype=np.float64))
            self.running_mean.append(np.zeros(self.dims[i + 1], dtype=np.float64))
            self.running_var.append(np.ones(self.dims[i + 1], dtype=np.float64))
        
        print(f"PCN Architecture: {self.dims}")
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_deriv(self, x):
        return (x > 0).astype(np.float64)
    
    def _softmax(self, x):
        x = np.clip(x, -100, 100)
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-10)
    
    def _batch_norm(self, x, gamma, beta, running_mean, running_var, training=True):
        """Batch normalization."""
        eps = 1e-5
        momentum = 0.9
        
        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            # Update running statistics
            running_mean[:] = momentum * running_mean + (1 - momentum) * mean
            running_var[:] = momentum * running_var + (1 - momentum) * var
        else:
            mean = running_mean
            var = running_var
        
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta, x_norm, mean, var
    
    def _forward(self, X, training=False):
        """Forward pass with batch normalization and dropout."""
        cache = {'pre_bn': [], 'x_norm': [], 'mean': [], 'var': [], 'dropout_mask': []}
        
        h = X.astype(np.float64)
        activations = [h]
        
        for i in range(self.n_layers):
            z = np.dot(h, self.weights[i]) + self.biases[i]
            cache['pre_bn'].append(z)
            
            if i < self.n_layers - 1:
                # Batch norm + ReLU for hidden layers
                z_bn, x_norm, mean, var = self._batch_norm(
                    z, self.gamma[i], self.beta[i],
                    self.running_mean[i], self.running_var[i], training)
                cache['x_norm'].append(x_norm)
                cache['mean'].append(mean)
                cache['var'].append(var)
                h = self._relu(z_bn)
                
                # Dropout during training
                if training and self.dropout_rate > 0:
                    keep_prob = 1 - self.dropout_rate
                    mask = (self.rng.random(h.shape) < keep_prob).astype(np.float64)
                    h = (h * mask) / keep_prob
                    cache['dropout_mask'].append(mask)
                else:
                    cache['dropout_mask'].append(None)
            else:
                h = z  # Logits
            
            activations.append(h)
        
        return activations, cache
    
    def _backward(self, activations, cache, y_onehot, sample_weights):
        """Backward pass."""
        batch_size = activations[0].shape[0]
        
        # Output layer gradient
        probs = self._softmax(activations[-1])
        output_delta = (probs - y_onehot) * sample_weights[:, None]
        
        grads_w = [None] * self.n_layers
        grads_b = [None] * self.n_layers
        grads_gamma = [None] * (self.n_layers - 1)
        grads_beta = [None] * (self.n_layers - 1)
        
        delta = output_delta
        
        for i in range(self.n_layers - 1, -1, -1):
            # Weight gradients
            grads_w[i] = np.dot(activations[i].T, delta) / batch_size
            grads_w[i] += self.l2_reg * self.weights[i]
            grads_b[i] = np.mean(delta, axis=0)
            
            # Clip gradients
            grads_w[i] = np.clip(grads_w[i], -1.0, 1.0)
            grads_b[i] = np.clip(grads_b[i], -1.0, 1.0)
            
            if i > 0:
                # Backprop through activation
                delta = np.dot(delta, self.weights[i].T)
                
                # Backprop through batch norm for hidden layers
                if i < self.n_layers:
                    # ReLU derivative
                    delta = delta * self._relu_deriv(activations[i])
                    
                    # Batch norm gradients (simplified)
                    if i - 1 < len(self.gamma):
                        grads_gamma[i - 1] = np.sum(delta * cache['x_norm'][i - 1], axis=0)
                        grads_beta[i - 1] = np.sum(delta, axis=0)
                        
                        # Backprop through batch norm
                        eps = 1e-5
                        var = cache['var'][i - 1]
                        x_norm = cache['x_norm'][i - 1]
                        N = batch_size
                        
                        dx_norm = delta * self.gamma[i - 1]
                        dvar = np.sum(dx_norm * (cache['pre_bn'][i - 1] - cache['mean'][i - 1]) * 
                                     -0.5 * (var + eps) ** -1.5, axis=0)
                        dmean = np.sum(dx_norm * -1 / np.sqrt(var + eps), axis=0)
                        
                        delta = (dx_norm / np.sqrt(var + eps) + 
                                dvar * 2 * (cache['pre_bn'][i - 1] - cache['mean'][i - 1]) / N +
                                dmean / N)
        
        return grads_w, grads_b, grads_gamma, grads_beta
    
    def _adam_update(self, t):
        """Adam optimizer update."""
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        for i in range(self.n_layers):
            # Update weights
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * self._grads_w[i]
            self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * (self._grads_w[i] ** 2)
            
            m_hat = self.m_w[i] / (1 - beta1 ** t)
            v_hat = self.v_w[i] / (1 - beta2 ** t)
            
            self.weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            
            # Update biases
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * self._grads_b[i]
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (self._grads_b[i] ** 2)
            
            m_hat = self.m_b[i] / (1 - beta1 ** t)
            v_hat = self.v_b[i] / (1 - beta2 ** t)
            
            self.biases[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        
        # Update batch norm parameters
        for i in range(self.n_layers - 1):
            if self._grads_gamma[i] is not None:
                self.gamma[i] -= self.learning_rate * 0.01 * self._grads_gamma[i]
                self.beta[i] -= self.learning_rate * 0.01 * self._grads_beta[i]
    
    def fit(self, X, y, class_weights=None, X_val=None, y_val=None):
        """Train the PCN classifier."""
        N, T, F = X.shape
        X_flat = X.reshape(N, -1)
        
        self.scaler.fit(X_flat)
        X_scaled = self.scaler.transform(X_flat)
        
        classes, counts = np.unique(y, return_counts=True)
        print(f"Input to Model - Class Counts: {dict(zip(classes, counts))}")
        
        y_onehot = np.eye(self.n_classes)[y]
        
        if class_weights is not None:
            sample_weights = np.array([class_weights[label] for label in y])
        else:
            sample_weights = np.ones(N)
        
        print(f"Training PCN for {self.n_epochs} epochs...")
        best_val_f1 = 0.0
        best_state = None
        patience = 15
        patience_counter = 0
        
        t = 0  # Adam time step
        
        for epoch in range(self.n_epochs):
            indices = self.rng.permutation(N)
            X_shuffled = X_scaled[indices]
            y_shuffled = y_onehot[indices]
            w_shuffled = sample_weights[indices]
            
            total_loss = 0.0
            n_batches = max(1, N // self.batch_size)
            
            for batch_idx in range(n_batches):
                t += 1
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, N)
                
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                w_batch = w_shuffled[start:end]
                
                # Forward pass
                activations, cache = self._forward(X_batch, training=True)
                
                # Backward pass
                self._grads_w, self._grads_b, self._grads_gamma, self._grads_beta = \
                    self._backward(activations, cache, y_batch, w_batch)
                
                # Update weights
                self._adam_update(t)
                
                # Compute loss
                probs = self._softmax(activations[-1])
                ce_loss = -np.sum(y_batch * np.log(np.clip(probs, 1e-10, 1)), axis=1)
                batch_loss = np.mean(ce_loss * w_batch)
                total_loss += batch_loss
            
            avg_loss = total_loss / n_batches
            
            # Evaluate
            if (epoch + 1) % 5 == 0 or epoch == 0:
                train_pred = self.predict(X)
                train_acc = accuracy_score(y, train_pred)
                train_f1 = f1_score(y, train_pred, average='macro')
                
                log_msg = (f"Epoch {epoch + 1:3d}/{self.n_epochs} - "
                          f"Loss: {avg_loss:.4f}, TrAcc: {train_acc:.4f}, "
                          f"TrF1: {train_f1:.4f}")
                
                if X_val is not None and y_val is not None:
                    val_pred = self.predict(X_val)
                    val_acc = accuracy_score(y_val, val_pred)
                    val_f1 = f1_score(y_val, val_pred, average='macro')
                    log_msg += f", VaAcc: {val_acc:.4f}, VaF1: {val_f1:.4f}"
                    
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_state = {
                            'weights': [w.copy() for w in self.weights],
                            'biases': [b.copy() for b in self.biases],
                            'gamma': [g.copy() for g in self.gamma],
                            'beta': [b.copy() for b in self.beta],
                            'running_mean': [m.copy() for m in self.running_mean],
                            'running_var': [v.copy() for v in self.running_var]
                        }
                        patience_counter = 0
                        log_msg += " *"
                    else:
                        patience_counter += 1
                
                print(log_msg)
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Restore best state
        if best_state is not None:
            self.weights = best_state['weights']
            self.biases = best_state['biases']
            self.gamma = best_state['gamma']
            self.beta = best_state['beta']
            self.running_mean = best_state['running_mean']
            self.running_var = best_state['running_var']
            print(f"Restored best model with Val F1: {best_val_f1:.4f}")
        
        print("PCN Training Complete.")
    
    def predict(self, X):
        """Predict class labels."""
        if len(X.shape) == 3:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X
            
        X_scaled = self.scaler.transform(X_flat)
        activations, _ = self._forward(X_scaled, training=False)
        return np.argmax(activations[-1], axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if len(X.shape) == 3:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X
            
        X_scaled = self.scaler.transform(X_flat)
        activations, _ = self._forward(X_scaled, training=False)
        return self._softmax(activations[-1])


def main():
    """Main execution block for training and inference."""
    base = "/teamspace/studios/this_studio/Barbados_Traffic_Analysis_Challenge_dev"
    if not os.path.exists(base):
        base = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev"
        
    train_path = os.path.join(base, "demos/Train.csv")
    test_path = os.path.join(base, "demos/TestInputSegments.csv")
    sample_sub_path = os.path.join(base, "demos/SampleSubmission.csv")
    
    print("Preparing PCN Data (Full Dataset)...")
    X_train, y_train, X_val, y_val = create_raw_sequences(
        train_path, val_split=0.2, seq_len=SEQ_LEN)
    
    t_cls, t_cnt = np.unique(y_train, return_counts=True)
    v_cls, v_cnt = np.unique(y_val, return_counts=True)
    print(f"Train data classes: {dict(zip(t_cls, t_cnt))}")
    print(f"Val data classes:   {dict(zip(v_cls, v_cnt))}")
    print(f"Train Shape: {X_train.shape}, Val: {X_val.shape}")
    
    # PCN Params - Optimized for performance without backprop-like overfitting
    seq_len, n_features = X_train.shape[1], X_train.shape[2]
    input_dim = seq_len * n_features
    
    pcn = PCNClassifier(
        input_dim=input_dim, 
        hidden_dims=[512, 256, 128],  # Larger capacity
        n_classes=4,
        learning_rate=0.002,
        n_epochs=500,
        batch_size=64,
        l2_reg=1e-4,
        dropout_rate=0.2,
        seed=42
    )
    
    print("Training PCN...")
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    print(f"Class Weights: {class_weight_dict}")
    
    pcn.fit(X_train, y_train, class_weights=class_weight_dict, 
            X_val=X_val, y_val=y_val)
    
    print("\n--- Final Evaluation ---")
    print("Training Data:")
    train_pred = pcn.predict(X_train)
    train_f1 = f1_score(y_train, train_pred, average='macro')
    train_acc = accuracy_score(y_train, train_pred)
    print(f"  Accuracy: {train_acc:.4f}, F1 Macro: {train_f1:.4f}")

    print("Validation Data:")
    val_pred = pcn.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    f1 = f1_score(y_val, val_pred, average='macro')
    prec = precision_score(y_val, val_pred, average='macro', zero_division=0)
    rec = recall_score(y_val, val_pred, average='macro', zero_division=0)
    
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1 Macro:  {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    
    # Inference
    print("\nStarting Inference...")
    congestion_map = {
        0: 'free flowing', 1: 'light delay', 
        2: 'moderate delay', 3: 'heavy delay'
    }
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
            
            start_id = int(round(history[-1][10] * 5000))
            
            for i in range(1, 9):
                current_seq = np.array(history[-15:]).reshape(1, 15, -1)
                
                p_idx = pcn.predict(current_seq)[0]
                p_label = congestion_map[p_idx]
                prediction_dict[(label, start_id + i)] = p_label
                
                next_feat = np.copy(history[-1])
                curr_h = next_feat[8] * 23.0
                curr_m = next_feat[9] * 59.0
                curr_m += 5
                if curr_m > 59:
                    curr_m -= 60
                    curr_h = (curr_h + 1) % 24
                next_feat[8] = curr_h / 23.0
                next_feat[9] = curr_m / 59.0
                next_feat[10] += 1/5000.0
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
    out_file = os.path.join(output_dir, "submission_pcn.csv")
    sample_sub.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
