"""Module to perform hyperparameter tuning for the Forward-Forward algorithm using Hyperband."""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import ops
import keras_tuner as kt
from tensorflow.compiler.tf2xla.python import xla
from sklearn.metrics import accuracy_score

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
keras.utils.set_random_seed(SEED)

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
    ], axis=1).astype('float32')
    
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
    def __init__(self, units, num_epochs=60, threshold=1.5, learning_rate=0.003, kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_epochs = num_epochs
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.kernel_regularizer = kernel_regularizer
        self.dense = keras.layers.Dense(
            units=units, 
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=keras.initializers.GlorotUniform(seed=SEED)
        )
        self.relu = keras.layers.ReLU()
        self.optimizer = keras.optimizers.Adam(learning_rate)
        self.loss_metric = keras.metrics.Mean()

    def call(self, x):
        x_norm = ops.norm(x, ord=2, axis=1, keepdims=True) + 1e-4
        return self.relu(self.dense(x / x_norm))

    def forward_forward(self, x_pos, x_neg, sample_weight=None):
        for i in range(self.num_epochs):
            with tf.GradientTape() as tape:
                g_pos = ops.mean(ops.power(self.call(x_pos), 2), 1)
                g_neg = ops.mean(ops.power(self.call(x_neg), 2), 1)
                
                loss_pos = ops.log(1 + ops.exp(-g_pos + self.threshold))
                loss_neg = ops.log(1 + ops.exp(g_neg - self.threshold))
                
                if sample_weight is not None:
                    loss_pos = loss_pos * ops.cast(sample_weight, loss_pos.dtype)
                
                loss = ops.concatenate([loss_pos, loss_neg], 0)
                mean_loss = ops.cast(ops.mean(loss), dtype="float32")
                if self.dense.losses:
                    mean_loss += ops.sum(self.dense.losses)
                self.loss_metric.update_state([mean_loss])
            grads = tape.gradient(mean_loss, self.dense.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.dense.trainable_weights))
        return ops.stop_gradient(self.call(x_pos)), ops.stop_gradient(self.call(x_neg)), self.loss_metric.result()

class FFNetwork(keras.Model):
    def __init__(self, dims, layer_epochs=60, threshold=1.5, learning_rate=0.003, kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.loss_var = keras.Variable(0.0, trainable=False)
        self.loss_count = keras.Variable(0.0, trainable=False)
        self.ff_layers = [
            FFDense(d, num_epochs=layer_epochs, threshold=threshold, learning_rate=learning_rate, kernel_regularizer=kernel_regularizer) 
            for d in dims[1:]
        ]
        self.acc_tracker = keras.metrics.SparseCategoricalAccuracy(name="acc")
        self.f1_tracker = keras.metrics.F1Score(name="f1", average="macro")

    @property
    def metrics(self):
        return [self.acc_tracker, self.f1_tracker]

    def overlay_y_on_x(self, data):
        x, y = data
        x_zeros = ops.zeros([16], dtype=x.dtype)
        update = xla.dynamic_update_slice(x_zeros, [ops.cast(1.0, x.dtype)], [y])
        return xla.dynamic_update_slice(x, update, [0]), y

    @tf.function
    def predict_batch(self, x):
        return ops.vectorized_map(self.predict_one, x)

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None

        x_pos, _ = ops.vectorized_map(self.overlay_y_on_x, (x, y))
        indices = tf.range(start=0, limit=tf.shape(y)[0], dtype=tf.int32)
        shuffled_indices = tf.random.experimental.stateless_shuffle(indices, seed=[SEED, SEED])
        x_neg, _ = ops.vectorized_map(self.overlay_y_on_x, (x, tf.gather(y, shuffled_indices)))
        
        self.loss_var.assign(0.0)
        self.loss_count.assign(0.0)
        
        h_pos, h_neg = x_pos, x_neg
        for layer in self.ff_layers:
            h_pos, h_neg, loss = layer.forward_forward(h_pos, h_neg, sample_weight=sample_weight)
            self.loss_var.assign_add(loss)
            self.loss_count.assign_add(1.0)
        
        y_pred = self.predict_batch(x)
        y_pred_1hot = ops.one_hot(y_pred, 16)
        self.acc_tracker.update_state(y, y_pred_1hot)
        self.f1_tracker.update_state(ops.one_hot(y, 16), y_pred_1hot)
        
        return {
            "loss": self.loss_var / self.loss_count, 
            "acc": self.acc_tracker.result(),
            "f1": self.f1_tracker.result()
        }

    def test_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data
            sample_weight = None
        
        x_pos, _ = ops.vectorized_map(self.overlay_y_on_x, (x, y))
        indices = tf.range(start=0, limit=tf.shape(y)[0], dtype=tf.int32)
        shuffled_indices = tf.random.experimental.stateless_shuffle(indices, seed=[SEED, SEED])
        x_neg, _ = ops.vectorized_map(self.overlay_y_on_x, (x, tf.gather(y, shuffled_indices)))
        
        v_loss = 0.0
        h_pos, h_neg = x_pos, x_neg
        for layer in self.ff_layers:
            g_pos = ops.mean(ops.power(layer(h_pos), 2), 1)
            g_neg = ops.mean(ops.power(layer(h_neg), 2), 1)
            loss_pos = ops.log(1 + ops.exp(-g_pos + layer.threshold))
            loss_neg = ops.log(1 + ops.exp(g_neg - layer.threshold))
            if sample_weight is not None:
                loss_pos = loss_pos * ops.cast(sample_weight, loss_pos.dtype)
            layer_loss = ops.mean(ops.concatenate([loss_pos, loss_neg], 0))
            v_loss += ops.cast(layer_loss, "float32")
            h_pos = ops.stop_gradient(layer(h_pos))
            h_neg = ops.stop_gradient(layer(h_neg))
        
        v_loss /= len(self.ff_layers)
        y_pred = self.predict_batch(x)
        y_pred_1hot = ops.one_hot(y_pred, 16)
        self.acc_tracker.update_state(y, y_pred_1hot)
        self.f1_tracker.update_state(ops.one_hot(y, 16), y_pred_1hot)
        return {"loss": v_loss, "acc": self.acc_tracker.result(), "f1": self.f1_tracker.result()}

    def predict_one(self, x):
        h_all = []
        for label in range(16):
            h, _ = self.overlay_y_on_x((x, label))
            h_all.append(h)
        h_all = ops.stack(h_all)
        h = h_all
        goodness = []
        for layer in self.ff_layers:
            h = layer(h)
            goodness.append(ops.mean(ops.power(h, 2), 1))
        total_goodness = ops.sum(ops.stack(goodness), 0)
        return ops.cast(ops.argmax(total_goodness), "int32")

# --- Keras Tuner Integration ---

def build_model(hp):
    num_layers = hp.Int("num_layers", 1, 5)
    units = hp.Choice("units", [256, 512, 768, 1024])
    learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
    threshold = hp.Float("threshold", 1.0, 3.0)
    layer_epochs = hp.Int("layer_epochs", 30, 100)
    l2_reg = hp.Float("l2_reg", 1e-5, 1e-2, sampling="log")
    
    dims = [25] + [units] * num_layers
    reg = keras.regularizers.L2(l2_reg)
    
    model = FFNetwork(
        dims=dims,
        layer_epochs=layer_epochs,
        threshold=threshold,
        learning_rate=learning_rate,
        kernel_regularizer=reg
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        jit_compile=False,
        metrics=["acc", "f1"]
    )
    return model

def main():
    base = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev"
    train_path = os.path.join(base, "demos/Train.csv")
    
    X, y = create_sequential_dataset(train_path)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(2048)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(2048)
    
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))
    cw_dict = {i: len(y_train) / (16.0 * class_counts.get(i, 1.0)) for i in range(16)}
    avg_w = sum(cw_dict.values()) / 16.0
    cw_dict = {k: v / avg_w for k, v in cw_dict.items()}

    tuner = kt.Hyperband(
        build_model,
        objective=kt.Objective("val_f1", direction="max"),
        max_epochs=20,
        factor=3,
        directory=os.path.join(base, "hyperband_search"),
        project_name="ff_traffic",
        seed=SEED
    )

    tuner.search(
        train_dataset,
        validation_data=val_dataset,
        class_weight=cw_dict,
        callbacks=[keras.callbacks.EarlyStopping(patience=3)]
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\nBest Hyperparameters:")
    print(f"Layers: {best_hps.get('num_layers')}")
    print(f"Units: {best_hps.get('units')}")
    print(f"Learning Rate: {best_hps.get('learning_rate')}")
    print(f"Threshold: {best_hps.get('threshold')}")
    print(f"Layer Epochs: {best_hps.get('layer_epochs')}")
    print(f"L2 Reg: {best_hps.get('l2_reg')}")

if __name__ == "__main__":
    main()
