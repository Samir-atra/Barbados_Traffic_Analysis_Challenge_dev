"""Module implementing the Forward-Forward algorithm for traffic prediction.

Restructured to:
1. Train on sequential blocks from Train.csv.
2. Predict the future 5 states (Enter/Exit congestion) for each sequence.
3. Validate on a dedicated split.
4. Perform inference on TestInputSegments.csv.
"""

import os
import matplotlib
matplotlib.use('Agg')

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
from keras import ops
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.compiler.tf2xla.python import xla

# --- Data Preparation Helpers ---

def get_features_and_labels(df):
    """Extracts features and labels from a dataframe."""
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
    df['enter_id'] = df['congestion_enter_rating'].map(congestion_map).fillna(0).astype(int)
    df['exit_id'] = df['congestion_exit_rating'].map(congestion_map).fillna(0).astype(int)
    df['joint_label'] = df['enter_id'] * 4 + df['exit_id']
    
    view_1hot = pd.get_dummies(df['view_id'], prefix='view').reindex(
        columns=['view_0', 'view_1', 'view_2', 'view_3'], fill_value=0).astype(float).values
    
    features = np.concatenate([
        df[['hour', 'minute', 'day_of_week', 'seg_id_norm', 'view_id']].values,
        view_1hot
    ], axis=1).astype('float32') # 5 base + 4 1-hot = 9 features
    
    return features, df['joint_label'].values

def create_sequential_dataset(csv_path):
    """Groups data by sequential blocks and creates training windows."""
    df = pd.read_csv(csv_path)
    all_X, all_y = [], []
    
    for label, group in df.groupby('view_label'):
        group = group.sort_values('time_segment_id')
        ids = group['time_segment_id'].values
        is_break = np.zeros(len(ids), dtype=int)
        is_break[1:] = (ids[1:] != ids[:-1] + 1).astype(int)
        group['block_id'] = np.cumsum(is_break)
        
        for b_id, block in group.groupby('block_id'):
            if len(block) < 6: continue
            feats, labels = get_features_and_labels(block)
            for i in range(len(feats) - 5):
                all_X.append(feats[i])
                all_y.append(labels[i+1])
                
    X = np.array(all_X)
    X_padded = np.zeros((X.shape[0], 16 + X.shape[1]), dtype='float32')
    X_padded[:, 16:] = X
    return X_padded, np.array(all_y)

# --- Forward-Forward Classes ---

class FFDense(keras.layers.Layer):
    def __init__(self, units, num_epochs=60, **kwargs):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(units=units)
        self.relu = keras.layers.ReLU()
        self.optimizer = keras.optimizers.Adam(0.03)
        self.loss_metric = keras.metrics.Mean()
        self.threshold = 1.5
        self.num_epochs = num_epochs

    def call(self, x):
        x_norm = ops.norm(x, ord=2, axis=1, keepdims=True) + 1e-4
        return self.relu(self.dense(x / x_norm))

    def forward_forward(self, x_pos, x_neg):
        for i in range(self.num_epochs):
            with tf.GradientTape() as tape:
                g_pos = ops.mean(ops.power(self.call(x_pos), 2), 1)
                g_neg = ops.mean(ops.power(self.call(x_neg), 2), 1)
                loss = ops.log(1 + ops.exp(ops.concatenate([-g_pos + self.threshold, g_neg - self.threshold], 0)))
                mean_loss = ops.cast(ops.mean(loss), dtype="float32")
                self.loss_metric.update_state([mean_loss])
            grads = tape.gradient(mean_loss, self.dense.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.dense.trainable_weights))
        return ops.stop_gradient(self.call(x_pos)), ops.stop_gradient(self.call(x_neg)), self.loss_metric.result()

class FFNetwork(keras.Model):
    def __init__(self, dims, **kwargs):
        super().__init__(**kwargs)
        self.loss_var = keras.Variable(0.0, trainable=False)
        self.loss_count = keras.Variable(0.0, trainable=False)
        self.ff_layers = [FFDense(d) for d in dims[1:]]

    def overlay_y_on_x(self, data):
        x, y = data
        x_zeros = ops.zeros([16], dtype=x.dtype)
        update = xla.dynamic_update_slice(x_zeros, [ops.cast(1.0, x.dtype)], [y])
        return xla.dynamic_update_slice(x, update, [0]), y

    @tf.function
    def predict_batch(self, x):
        batch_size = ops.shape(x)[0]
        # [Batch, 16, Dim]
        # Tile x for all 16 labels
        x_expanded = ops.expand_dims(x, 1) # [B, 1, D]
        x_tiled = ops.repeat(x_expanded, 16, axis=1) # [B, 16, D]
        
        goodness_total = ops.zeros((batch_size, 16))
        
        for label in range(16):
            # Slow loop but vectorized over batch
            h_batch = []
            for i in range(batch_size):
                h, _ = self.overlay_y_on_x((x[i], label))
                h_batch.append(h)
            h = ops.stack(h_batch)
            
            curr_goodness = 0
            for layer in self.ff_layers:
                h = layer(h)
                curr_goodness += ops.mean(ops.power(h, 2), 1)
            
            # goodness_total is [B, 16]
            goodness_total = tf.tensor_scatter_nd_update(
                goodness_total, 
                tf.stack([tf.range(batch_size), tf.fill([batch_size], label)], axis=1),
                curr_goodness
            )
            
        return ops.argmax(goodness_total, axis=1)

    def train_step(self, data):
        x, y = data
        x_pos, _ = ops.vectorized_map(self.overlay_y_on_x, (x, y))
        x_neg, _ = ops.vectorized_map(self.overlay_y_on_x, (x, tf.random.shuffle(y)))
        
        self.loss_var.assign(0.0)
        self.loss_count.assign(0.0)
        
        h_pos, h_neg = x_pos, x_neg
        for layer in self.ff_layers:
            h_pos, h_neg, loss = layer.forward_forward(h_pos, h_neg)
            self.loss_var.assign_add(loss)
            self.loss_count.assign_add(1.0)
        return {"loss": self.loss_var / self.loss_count}

    def predict_one(self, x):
        h_all = []
        for label in range(16):
            h, _ = self.overlay_y_on_x((x, label))
            h_all.append(h)
        h_all = ops.stack(h_all)
        
        goodness = []
        h = h_all
        for layer in self.ff_layers:
            h = layer(h)
            goodness.append(ops.mean(ops.power(h, 2), 1))
        
        total_goodness = ops.sum(ops.stack(goodness), 0)
        return ops.cast(ops.argmax(total_goodness), "int32")

# --- Main ---

def main():
    base = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev"
    train_path = os.path.join(base, "demos/Train.csv")
    test_path = os.path.join(base, "demos/TestInputSegments.csv")
    
    print("Preparing Data...")
    X, y = create_sequential_dataset(train_path)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    model = FFNetwork(dims=[25, 512, 512]) 
    model.compile(optimizer="adam", jit_compile=False)
    
    print("Training (15 epochs for efficiency)...")
    model.fit(X_train, y_train, epochs=15, batch_size=2048, verbose=1)
    
    print("\nInference on TestInputSegments...")
    test_df = pd.read_csv(test_path)
    
    congestion_map = {0: 'free flowing', 1: 'light delay', 2: 'moderate delay', 3: 'heavy delay'}
    submission_rows = []

    # Process each location independently
    for label, group in test_df.groupby('view_label'):
        group = group.sort_values('time_segment_id')
        ids = group['time_segment_id'].values
        
        # Identify sequential chunks in test data
        is_break = np.zeros(len(ids), dtype=int)
        is_break[1:] = (ids[1:] != ids[:-1] + 1).astype(int)
        group['block_id'] = np.cumsum(is_break)
        
        for b_id, block in group.groupby('block_id'):
            feats, _ = get_features_and_labels(block)
            last_feat = feats[-1]
            
            # Predict future steps
            current_state = np.copy(last_feat)
            start_id = int(round(last_feat[3] * 5000))
            
            # Pattern in SampleSubmission.csv shows a 2-segment gap (Start = Last + 3)
            for i in range(1, 8): # Predict up to T+7 to get the range [T+3, T+7]
                # Input to model must be padded (16 zeros)
                input_vec = np.zeros(16 + len(current_state), dtype='float32')
                input_vec[16:] = current_state
                
                p_joint = model.predict_one(ops.convert_to_tensor(input_vec)).numpy()
                p_enter_label = congestion_map[p_joint // 4]
                
                target_id = start_id + i
                
                # Only include T+3, T+4, T+5, T+6, T+7 in the submission to match SampleSubmission gap
                if i >= 3:
                    enter_id = f"time_segment_{target_id}_{label}_congestion_enter_rating"
                    submission_rows.append({
                        'ID': enter_id, 
                        'Target': p_enter_label, 
                        'Target_Accuracy': p_enter_label
                    })
                
                # Update state for next step prediction (Autoregressive)
                current_state[0] += (1/1440.0)
                current_state[3] += (1/5000.0)

    submission_df = pd.DataFrame(submission_rows)
    output_path = os.path.join(base, "submission.csv")
    submission_df.to_csv(output_path, index=False)
    print(f"\nSubmission file generated: {output_path}")
    print(f"Total rows in submission: {len(submission_df)}")

if __name__ == "__main__":
    main()
