"""Module to perform hyperparameter tuning for the Forward-Forward algorithm using Hyperband on Kaggle.

Optimized version to avoid OOM by using direct vectorization and respecting the sequential 15-step context window.
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import ops
import keras_tuner as kt

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
keras.utils.set_random_seed(SEED)

# --- Data Preparation Helpers ---

def get_features_and_labels(df):
    """Extracts and encodes 13 base features and labels from a traffic dataframe."""
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
    
    view_1hot = pd.get_dummies(df['view_id'], prefix='view').reindex(
        columns=['view_0', 'view_1', 'view_2', 'view_3'], fill_value=0).astype(float).values
    
    sig_map = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
    df['sig_id'] = df['signaling'].map(sig_map).fillna(0)
    sig_1hot = pd.get_dummies(df['sig_id'], prefix='sig').reindex(
        columns=['sig_0', 'sig_1', 'sig_2', 'sig_3'], fill_value=0).astype(float).values
    
    features = np.concatenate([
        df[['hour', 'minute', 'day_of_week', 'seg_id_norm', 'view_id']].values,
        view_1hot,
        sig_1hot
    ], axis=1).astype('float32') # 13 features
    
    return features, df['enter_id'].values

def create_sequential_dataset(csv_path, seq_len=15):
    """Groups data by sequential blocks and creates training windows of length seq_len."""
    df = pd.read_csv(csv_path)
    all_X, all_y = [], []
    
    for label, group in df.groupby('view_label'):
        group = group.sort_values('time_segment_id')
        
        # Identify non-contiguous sequential chunks
        ids = group['time_segment_id'].values
        is_break = np.zeros(len(ids), dtype=int)
        is_break[1:] = (ids[1:] != ids[:-1] + 1).astype(int)
        group['block_id'] = np.cumsum(is_break)
        
        for b_id, block in group.groupby('block_id'):
            if len(block) < seq_len + 1: continue
            feats, labels = get_features_and_labels(block)
            for i in range(len(feats) - seq_len):
                # Flatten window of features: 15 * 13 = 195 features
                window = feats[i : i + seq_len].flatten()
                all_X.append(window)
                all_y.append(labels[i + seq_len])
                
    X = np.array(all_X)
    # Total input: 4 (label buffer) + 195 (features) = 199
    X_padded = np.zeros((X.shape[0], 4 + X.shape[1]), dtype='float32')
    X_padded[:, 4:] = X
    return X_padded, np.array(all_y)

# --- Forward-Forward Classes ---

class FFDense(keras.layers.Layer):
    """A single Forward-Forward Dense layer with local learning and EMA."""
    def __init__(self, units, num_epochs=54, threshold=1.5, learning_rate=0.001, 
                 gamma=2.0, use_ema=True, ema_overwrite_frequency=2, kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_epochs = num_epochs
        self.threshold = threshold
        self.gamma = gamma
        self.dense = keras.layers.Dense(
            units=units, 
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=keras.initializers.GlorotUniform(seed=SEED)
        )
        self.ln = keras.layers.LayerNormalization()
        self.activation = keras.layers.LeakyReLU(negative_slope=0.2)
        # Apply EMA and Clipping directly to the layer-wise optimizer
        self.optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate, 
            global_clipnorm=1.0, 
            use_ema=use_ema,
            ema_overwrite_frequency=ema_overwrite_frequency
        )
        self.loss_metric = keras.metrics.Mean()

    def call(self, x):
        x_norm = ops.norm(x, ord=2, axis=1, keepdims=True) + 1e-4
        h = self.dense(x / x_norm)
        return self.activation(self.ln(h))

    def forward_forward(self, x_pos, x_neg):
        """Local training using Focal Loss weighting."""
        for i in range(self.num_epochs):
            with tf.GradientTape() as tape:
                g_pos = ops.mean(ops.power(self.call(x_pos), 2), 1)
                g_neg = ops.mean(ops.power(self.call(x_neg), 2), 1)
                
                log_pos = ops.log(1 + ops.exp(-g_pos + self.threshold))
                log_neg = ops.log(1 + ops.exp(g_neg - self.threshold))
                
                pt_pos = ops.sigmoid(-g_pos + self.threshold)
                pt_neg = ops.sigmoid(g_neg - self.threshold)
                
                loss_pos = ops.power(pt_pos, self.gamma) * log_pos
                loss_neg = ops.power(pt_neg, self.gamma) * log_neg
                
                loss = ops.concatenate([loss_pos, loss_neg], 0)
                mean_loss = ops.cast(ops.mean(loss), dtype="float32")
                if self.dense.losses:
                    mean_loss += ops.sum(self.dense.losses)
                self.loss_metric.update_state([mean_loss])
            grads = tape.gradient(mean_loss, self.dense.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.dense.trainable_weights))
        return ops.stop_gradient(self.call(x_pos)), ops.stop_gradient(self.call(x_neg)), self.loss_metric.result()

class FFNetwork(keras.Model):
    """Full Forward-Forward network with optimized batch vectorization."""
    def __init__(self, dims, layer_epochs=54, threshold=1.5, learning_rate=0.001, 
                 gamma=2.0, use_ema=True, ema_overwrite_frequency=2, kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.loss_var = keras.Variable(0.0, trainable=False)
        self.loss_count = keras.Variable(0.0, trainable=False)
        self.ff_layers = [
            FFDense(d, num_epochs=layer_epochs, threshold=threshold, 
                    learning_rate=learning_rate, gamma=gamma, 
                    use_ema=use_ema, ema_overwrite_frequency=ema_overwrite_frequency,
                    kernel_regularizer=kernel_regularizer) 
            for d in dims[1:]
        ]
        self.acc_tracker = keras.metrics.SparseCategoricalAccuracy(name="acc")
        self.f1_tracker = keras.metrics.F1Score(name="f1", average="macro")
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.build((None, dims[0]))

    @property
    def metrics(self):
        return [self.acc_tracker, self.f1_tracker, self.loss_tracker]

    def call(self, x):
        h = x
        for layer in self.ff_layers:
            h = layer(h)
        return h

    def predict_batch(self, x):
        """Vectorized batch prediction for multi-step feature windows."""
        batch_size = ops.shape(x)[0]
        # Features are at indices [4:], length is 195 (seq_len=15 * 13)
        x_features = x[:, 4:] 
        
        # Expand features to (B, 4, 195)
        x_expanded = ops.repeat(ops.expand_dims(x_features, 1), 4, axis=1)
        
        # Create all 4 labels for all samples
        labels = ops.arange(4)
        labels_1hot = ops.one_hot(labels, 4) # (4, 4)
        labels_expanded = ops.repeat(ops.expand_dims(labels_1hot, 0), batch_size, axis=0) # (B, 4, 4)
        
        # Concatenate to (B, 4, 199) and flatten to (B*4, 199)
        h = ops.concatenate([labels_expanded, x_expanded], axis=2)
        h = ops.reshape(h, (batch_size * 4, 199))
        
        goodness_sum = ops.zeros((batch_size * 4,))
        for layer in self.ff_layers:
            h = layer(h)
            goodness_sum += ops.mean(ops.power(h, 2), 1)
            
        goodness_sum = ops.reshape(goodness_sum, (batch_size, 4))
        return ops.argmax(goodness_sum, axis=1)

    def train_step(self, data):
        x, y = data
        
        # Positive samples: actual label overlay (batch, 199)
        y_1hot = ops.one_hot(y, 4)
        x_pos = ops.concatenate([y_1hot, x[:, 4:]], axis=1)
        
        # Negative samples: shuffled label overlay
        indices = tf.range(start=0, limit=tf.shape(y)[0], dtype=tf.int32)
        shuffled_indices = tf.random.experimental.stateless_shuffle(indices, seed=[SEED, SEED])
        y_neg_1hot = ops.one_hot(tf.gather(y, shuffled_indices), 4)
        x_neg = ops.concatenate([y_neg_1hot, x[:, 4:]], axis=1)
        
        self.loss_var.assign(0.0)
        self.loss_count.assign(0.0)
        
        h_pos, h_neg = x_pos, x_neg
        for layer in self.ff_layers:
            h_pos, h_neg, loss = layer.forward_forward(h_pos, h_neg)
            self.loss_var.assign_add(loss)
            self.loss_count.assign_add(1.0)
        
        self.loss_tracker.update_state(self.loss_var / self.loss_count)
        
        # Metrics update (batch)
        y_pred = self.predict_batch(x)
        y_pred_1hot = ops.one_hot(y_pred, 4)
        self.acc_tracker.update_state(y, y_pred_1hot)
        self.f1_tracker.update_state(ops.one_hot(y, 4), y_pred_1hot)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        
        # Calculate goodness-based loss for validation
        y_1hot = ops.one_hot(y, 4)
        x_pos = ops.concatenate([y_1hot, x[:, 4:]], axis=1)
        
        indices = tf.range(start=0, limit=tf.shape(y)[0], dtype=tf.int32)
        shuffled_indices = tf.random.experimental.stateless_shuffle(indices, seed=[SEED, SEED])
        y_neg_1hot = ops.one_hot(tf.gather(y, shuffled_indices), 4)
        x_neg = ops.concatenate([y_neg_1hot, x[:, 4:]], axis=1)
        
        v_loss = 0.0
        h_pos, h_neg = x_pos, x_neg
        for layer in self.ff_layers:
            g_pos = ops.mean(ops.power(layer(h_pos), 2), 1)
            g_neg = ops.mean(ops.power(layer(h_neg), 2), 1)
            
            pt_pos = ops.sigmoid(-g_pos + layer.threshold)
            pt_neg = ops.sigmoid(g_neg - layer.threshold)
            
            log_pos = ops.log(1 + ops.exp(-g_pos + layer.threshold))
            log_neg = ops.log(1 + ops.exp(g_neg - layer.threshold))
            
            layer_loss = ops.mean(ops.concatenate([
                ops.power(pt_pos, layer.gamma) * log_pos, 
                ops.power(pt_neg, layer.gamma) * log_neg
            ], 0))
            
            v_loss += ops.cast(layer_loss, "float32")
            h_pos = ops.stop_gradient(layer(h_pos))
            h_neg = ops.stop_gradient(layer(h_neg))
        
        self.loss_tracker.update_state(v_loss / len(self.ff_layers))
        
        y_pred = self.predict_batch(x)
        y_pred_1hot = ops.one_hot(y_pred, 4)
        self.acc_tracker.update_state(y, y_pred_1hot)
        self.f1_tracker.update_state(ops.one_hot(y, 4), y_pred_1hot)
        return {m.name: m.result() for m in self.metrics}

# --- Keras Tuner Integration ---

def build_model(hp):
    # Wider search ranges as requested
    num_layers = hp.Int("num_layers", 2, 5) 
    units = hp.Choice("units", [128, 256, 512, 768])
    base_lr = hp.Float("learning_rate", 5e-5, 2e-3, sampling="log")
    threshold = hp.Float("threshold", 0.8, 2.0)
    gamma = hp.Float("gamma", 0.5, 3.0)
    layer_epochs = hp.Int("layer_epochs", 30, 120)
    l2_reg = hp.Float("l2_reg", 1e-6, 1e-3, sampling="log")
    ema_freq = hp.Int("ema_overwrite_frequency", 1, 10)
    
    # Calculate Total Steps for Cosine Decay
    # We use a batch size of 256 for the tuner
    global_max_epochs = 12 # Hyperband max
    steps_per_epoch = 12000 // 256 # Approx train size
    total_train_steps = global_max_epochs * steps_per_epoch * layer_epochs
    
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=base_lr,
        decay_steps=total_train_steps,
        alpha=0.1
    )
    
    dims = [199] + [units] * num_layers
    reg = keras.regularizers.L2(l2_reg)
    
    model = FFNetwork(
        dims=dims,
        layer_epochs=layer_epochs,
        threshold=threshold,
        learning_rate=lr_schedule,
        gamma=gamma,
        use_ema=True,
        ema_overwrite_frequency=ema_freq,
        kernel_regularizer=reg
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(lr_schedule, global_clipnorm=1.0),
        jit_compile=False,
        metrics=["acc", "f1"]
    )
    return model

def main():
    train_path = "/kaggle/input/barbados-traffic-analysis-challenge/Train.csv"
    if not os.path.exists(train_path):
        train_path = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev/demos/Train.csv"
    
    print("Preparing Data (Sequence Length 15)...")
    X, y = create_sequential_dataset(train_path, seq_len=15)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Sequential order training (no balanced sampling as requested)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(256).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(256)

    tuner = kt.Hyperband(
        build_model,
        objective=kt.Objective("val_f1", direction="max"),
        max_epochs=12,
        factor=3,
        directory="hyperband_search",
        project_name="ff_traffic_sequential_v1",
        seed=SEED
    )

    tuner.search(
        train_dataset,
        validation_data=val_dataset,
        callbacks=[keras.callbacks.EarlyStopping(patience=3)]
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nBest Hyperparameters Found:")
    for key, value in best_hps.values.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
