"""Module implementing the Forward-Forward algorithm for traffic prediction (Kaggle TPU Version).

This script implements a Forward-Forward (FF) neural network optimized for TPU execution
to predict future traffic congestion states based on sequential historical data.

Workflow:
1. Initialize TPU and distribution strategy.
2. Load and preprocess traffic data from Train.csv.
3. Group data into sequential blocks for time-series prediction.
4. Train a Forward-Forward network with Focal Loss using TPU acceleration.
5. Predict the next 8 congestion states for TestInputSegments.csv.
6. Generate a submission-ready CSV file.
"""

import os
import matplotlib
matplotlib.use('Agg')

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
from keras import ops
import numpy as np
import polars as pl
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import random

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
keras.utils.set_random_seed(SEED)

# --- TPU Initialization ---
print("Detecting TPU...")
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print(f"TPU initialized: {resolver.cluster_spec().as_dict()}")
    print(f"Number of replicas: {strategy.num_replicas_in_sync}")
except ValueError:
    print("TPU not found, falling back to default strategy")
    strategy = tf.distribute.get_strategy()

# --- Data Preparation Helpers ---

def get_features_and_labels(df):
    """Extracts and encodes features and labels from a traffic dataframe (Polars + JAX)."""
    df = df.with_columns([
        pl.col("video_time").str.to_datetime(),
        pl.col("date").str.to_datetime()
    ])

    df = df.with_columns([
        (pl.col("video_time").dt.hour() / 23.0).alias("hour"),
        (pl.col("video_time").dt.minute() / 59.0).alias("minute"),
        ((pl.col("date").dt.weekday() - 1) / 6.0).alias("day_of_week")
    ])
    
    view_map = {
        'Norman Niles #1': 0, 
        'Norman Niles #2': 1, 
        'Norman Niles #3': 2, 
        'Norman Niles #4': 3
    }
    df = df.with_columns(
        pl.col("view_label").replace(view_map, default=0).cast(pl.Int32).alias("view_id"),
        (pl.col("time_segment_id") / 5000.0).alias("seg_id_norm")
    )
    
    if "congestion_enter_rating" in df.columns:
        congestion_map = {
            'free flowing': 0,
            'light delay': 1,
            'moderate delay': 2,
            'heavy delay': 3
        }
        df = df.with_columns(
            pl.col("congestion_enter_rating").replace(congestion_map, default=0).cast(pl.Int32).alias("enter_id")
        )
        labels = df['enter_id'].to_numpy()
    else:
        labels = np.zeros(len(df), dtype=np.int32)
    
    view_ids = df["view_id"].to_numpy()
    view_ids_jax = jnp.array(view_ids)
    view_1hot = jnp.zeros((len(view_ids), 4), dtype=float)
    view_1hot = view_1hot.at[jnp.arange(len(view_ids)), view_ids_jax].set(1.0)
    
    sig_map = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
    df = df.with_columns(
         pl.col("signaling").replace(sig_map, default=0).cast(pl.Int32).alias("sig_id")
    )
    sig_ids = df["sig_id"].to_numpy()
    sig_ids_jax = jnp.array(sig_ids)
    sig_1hot = jnp.zeros((len(sig_ids), 4), dtype=float)
    sig_1hot = sig_1hot.at[jnp.arange(len(sig_ids)), sig_ids_jax].set(1.0)
    
    base_feats = jnp.array(df.select(["hour", "minute", "day_of_week", "seg_id_norm", "view_id"]).to_numpy())
    
    features = jnp.concatenate([
        base_feats,
        view_1hot,
        sig_1hot
    ], axis=1).astype('float32')
    
    return features, labels

def identify_blocks(group):
    """Sorts by time_segment_id and identifies continuous sequential blocks (Polars)."""
    group = group.sort("time_segment_id")
    diff = group["time_segment_id"].diff().fill_null(1)
    is_break = (diff != 1).cast(pl.Int32)
    group = group.with_columns(is_break.cum_sum().alias("block_id"))
    return group

def create_dataset_splits(csv_path, val_split=0.2):
    """Processes CSV data into sequential training and validation samples."""
    df = pl.read_csv(csv_path)
    train_X, train_y = [], []
    val_X, val_y = [], []
    seq_len = 15
    
    print(f"Loading {len(df)} rows from {csv_path} with Polars...")
    
    partitions = df.partition_by("view_label")
    
    for group in partitions:
        label = group["view_label"][0]
        print(f"  Processing view: {label}")
        
        group = identify_blocks(group)
        view_X, view_y = [], []
        
        block_partitions = group.partition_by("block_id")
        
        for block in block_partitions:
            if len(block) < seq_len + 1: continue
            feats, labels = get_features_and_labels(block)
            
            n_samples = len(feats) - seq_len
            if n_samples > 0:
                indexer = jnp.arange(seq_len)[None, :] + jnp.arange(n_samples)[:, None]
                windows = feats[indexer].reshape(n_samples, -1)
                targets = labels[seq_len:]
                
                view_X.extend(np.array(windows)) 
                view_y.extend(targets)
        
        n_total = len(view_X)
        if n_total > 0:
            n_val = int(n_total * val_split)
            n_train = n_total - n_val
            
            train_X.extend(view_X[:n_train])
            train_y.extend(view_y[:n_train])
            val_X.extend(view_X[n_train:])
            val_y.extend(view_y[n_train:])
                
    def pad_X(X_list):
        if not X_list: return jnp.zeros((0, 15*13 + 4), dtype='float32')
        X_arr = jnp.array(X_list)
        X_padded = jnp.zeros((X_arr.shape[0], 4 + X_arr.shape[1]), dtype='float32')
        X_padded = X_padded.at[:, 4:].set(X_arr)
        return X_padded

    return pad_X(train_X), np.array(train_y), pad_X(val_X), np.array(val_y)

# --- Forward-Forward Classes ---

@keras.saving.register_keras_serializable(package="MyLayers")
class FFDense(keras.layers.Layer):
    """A single Forward-Forward Dense layer with local learning."""

    def __init__(self, units, num_epochs=54, kernel_regularizer=None, gamma=1.0337, 
                 threshold=1.143, learning_rate=0.001, use_ema=True, ema_overwrite_frequency=1, 
                 activation='leaky_relu', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_epochs = num_epochs
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.gamma = gamma
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.use_ema = use_ema
        self.ema_overwrite_frequency = ema_overwrite_frequency
        self.activation_name = activation
        
        self.dense = keras.layers.Dense(
            units=units, 
            kernel_regularizer=self.kernel_regularizer,
            kernel_initializer=keras.initializers.GlorotUniform(seed=SEED)
        )
        if activation == 'leaky_relu':
            self.activation = keras.layers.LeakyReLU(negative_slope=0.2)
        else:
            self.activation = keras.layers.Activation(activation)
            
        self.optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate, 
            global_clipnorm=1.0, 
            use_ema=use_ema,
            ema_overwrite_frequency=ema_overwrite_frequency
        )
        self.loss_metric = keras.metrics.Mean()

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "num_epochs": self.num_epochs,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "gamma": self.gamma,
            "threshold": self.threshold,
            "learning_rate": self.learning_rate,
            "use_ema": self.use_ema,
            "ema_overwrite_frequency": self.ema_overwrite_frequency,
            "activation": self.activation_name,
        })
        return config

    def call(self, x):
        x_norm = ops.norm(x, ord=2, axis=1, keepdims=True) + 1e-4
        h = self.dense(x / x_norm)
        return self.activation(h)

    def forward_forward(self, x_pos, x_neg, weights=None):
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
                
                if weights is not None:
                    loss_pos = loss_pos * weights
                    loss_neg = loss_neg * weights
                
                loss = ops.concatenate([loss_pos, loss_neg], 0)
                mean_loss = ops.cast(ops.mean(loss), dtype="float32")
                if self.dense.losses:
                    mean_loss += ops.sum(self.dense.losses)
                self.loss_metric.update_state([mean_loss])
            grads = tape.gradient(mean_loss, self.dense.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.dense.trainable_weights))
        return ops.stop_gradient(self.call(x_pos)), ops.stop_gradient(self.call(x_neg)), self.loss_metric.result()

@keras.saving.register_keras_serializable(package="MyMetrics")
class MacroPrecision(keras.metrics.Metric):
    def __init__(self, num_classes=4, name="macro_precision", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros")
        self.fp = self.add_weight(name="fp", shape=(num_classes,), initializer="zeros")

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")
        tp = ops.sum(y_true * y_pred, axis=0)
        fp = ops.sum((1 - y_true) * y_pred, axis=0)
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)

    def result(self):
        precisions = self.tp / (self.tp + self.fp + 1e-7)
        return ops.mean(precisions)

    def reset_state(self):
        self.tp.assign(ops.zeros_like(self.tp))
        self.fp.assign(ops.zeros_like(self.fp))

@keras.saving.register_keras_serializable(package="MyMetrics")
class MacroRecall(keras.metrics.Metric):
    def __init__(self, num_classes=4, name="macro_recall", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros")
        self.fn = self.add_weight(name="fn", shape=(num_classes,), initializer="zeros")

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")
        tp = ops.sum(y_true * y_pred, axis=0)
        fn = ops.sum(y_true * (1 - y_pred), axis=0)
        self.tp.assign_add(tp)
        self.fn.assign_add(fn)

    def result(self):
        recalls = self.tp / (self.tp + self.fn + 1e-7)
        return ops.mean(recalls)

    def reset_state(self):
        self.tp.assign(ops.zeros_like(self.tp))
        self.fn.assign(ops.zeros_like(self.fn))

@keras.saving.register_keras_serializable(package="MyModels")
class FFNetwork(keras.Model):
    """The full Forward-Forward network model."""

    def __init__(self, dims, kernel_regularizer=None, learning_rate=0.001, 
                 use_ema=True, ema_overwrite_frequency=None, 
                 layer_epochs=54, threshold=1.143, gamma=1.0337, **kwargs):
        super().__init__(**kwargs)
        self.dims = dims
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.learning_rate = learning_rate
        self.use_ema = use_ema
        self.ema_overwrite_frequency = ema_overwrite_frequency
        self.layer_epochs = layer_epochs
        self.threshold = threshold
        self.gamma = gamma

        self.loss_var = keras.Variable(0.0, trainable=False)
        self.loss_count = keras.Variable(0.0, trainable=False)
        self.ff_layers = []
        for i, d in enumerate(dims[1:]):
            act = 'softmax' if i == len(dims[1:]) - 1 else 'leaky_relu'
            self.ff_layers.append(
                FFDense(d, kernel_regularizer=self.kernel_regularizer, 
                        learning_rate=learning_rate, use_ema=use_ema,
                        ema_overwrite_frequency=ema_overwrite_frequency,
                        num_epochs=layer_epochs, threshold=threshold, gamma=gamma,
                        activation=act)
            )
        self.acc_tracker = keras.metrics.SparseCategoricalAccuracy(name="acc")
        self.f1_tracker = keras.metrics.F1Score(name="f1", average="macro")
        self.precision_tracker = MacroPrecision(name="precision")
        self.recall_tracker = MacroRecall(name="recall")
        self.build((None, dims[0]))

    def get_config(self):
        config = super().get_config()
        config.update({
            "dims": self.dims,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "learning_rate": self.learning_rate,
            "use_ema": self.use_ema,
            "ema_overwrite_frequency": self.ema_overwrite_frequency,
            "layer_epochs": self.layer_epochs,
            "threshold": self.threshold,
            "gamma": self.gamma,
        })
        return config

    @property
    def metrics(self):
        return [self.acc_tracker, self.f1_tracker, self.precision_tracker, self.recall_tracker]

    def call(self, x):
        h = x
        for layer in self.ff_layers:
            h = layer(h)
        return h

    def overlay_y_on_x(self, data):
        x, y = data
        y = ops.cast(y, "int32")
        y_scalar = ops.reshape(y, [])
        header = ops.one_hot(y_scalar, 4) * ops.cast(10.0, x.dtype)
        x_rest = x[4:]
        x_new = ops.concatenate([header, x_rest], axis=0)
        return x_new, y

    @tf.function
    def predict_batch(self, x):
        return ops.vectorized_map(self.predict_one, x)

    def train_step(self, data):
        if len(data) == 3:
            x, y, weights = data
            weights = ops.cast(weights, "float32")
        else:
            x, y = data
            weights = None

        x_pos, _ = ops.vectorized_map(self.overlay_y_on_x, (x, y))
        
        batch_size = tf.shape(y)[0]
        offsets = tf.random.uniform(shape=[batch_size], minval=1, maxval=4, dtype=tf.int32)
        y_neg_labels = (tf.cast(y, tf.int32) + offsets) % 4
        x_neg, _ = ops.vectorized_map(self.overlay_y_on_x, (x, y_neg_labels))
        
        self.loss_var.assign(0.0)
        self.loss_count.assign(0.0)
        
        h_pos, h_neg = x_pos, x_neg
        for layer in self.ff_layers:
            h_pos, h_neg, loss = layer.forward_forward(h_pos, h_neg, weights=weights)
            self.loss_var.assign_add(loss)
            self.loss_count.assign_add(1.0)
        
        y_pred = self.predict_batch(x)
        y_pred_1hot = ops.one_hot(y_pred, 4)
        self.acc_tracker.update_state(y, y_pred_1hot)
        self.f1_tracker.update_state(ops.one_hot(y, 4), y_pred_1hot)
        self.precision_tracker.update_state(ops.one_hot(y, 4), y_pred_1hot)
        self.recall_tracker.update_state(ops.one_hot(y, 4), y_pred_1hot)
        
        return {
            "loss": self.loss_var / self.loss_count, 
            "acc": self.acc_tracker.result(),
            "f1": self.f1_tracker.result(),
            "precision": self.precision_tracker.result(),
            "recall": self.recall_tracker.result()
        }

    def test_step(self, data):
        if len(data) == 3:
            x, y, weights = data
            weights = ops.cast(weights, "float32")
        else:
            x, y = data
            weights = None
        
        x_pos, _ = ops.vectorized_map(self.overlay_y_on_x, (x, y))
        
        batch_size = tf.shape(y)[0]
        offsets = tf.random.stateless_uniform(shape=[batch_size], seed=[SEED, SEED], minval=1, maxval=4, dtype=tf.int32)
        y_neg_labels = (tf.cast(y, tf.int32) + offsets) % 4
        x_neg, _ = ops.vectorized_map(self.overlay_y_on_x, (x, y_neg_labels))
        
        v_loss = 0.0
        h_pos, h_neg = x_pos, x_neg
        for layer in self.ff_layers:
            g_pos = ops.mean(ops.power(layer(h_pos), 2), 1)
            g_neg = ops.mean(ops.power(layer(h_neg), 2), 1)
            
            log_pos = ops.log(1 + ops.exp(-g_pos + layer.threshold))
            log_neg = ops.log(1 + ops.exp(g_neg - layer.threshold))
            pt_pos = ops.sigmoid(-g_pos + layer.threshold)
            pt_neg = ops.sigmoid(g_neg - layer.threshold)
            
            layer_loss = ops.power(pt_pos, layer.gamma) * log_pos
            layer_loss_neg = ops.power(pt_neg, layer.gamma) * log_neg
            
            if weights is not None:
                layer_loss = layer_loss * weights
                layer_loss_neg = layer_loss_neg * weights
                
            layer_loss_mean = ops.mean(ops.concatenate([layer_loss, layer_loss_neg], 0))
            v_loss += ops.cast(layer_loss_mean, "float32")
            h_pos = ops.stop_gradient(layer(h_pos))
            h_neg = ops.stop_gradient(layer(h_neg))
        
        v_loss /= len(self.ff_layers)

        y_pred = self.predict_batch(x)
        y_pred_1hot = ops.one_hot(y_pred, 4)
        self.acc_tracker.update_state(y, y_pred_1hot)
        self.f1_tracker.update_state(ops.one_hot(y, 4), y_pred_1hot)
        self.precision_tracker.update_state(ops.one_hot(y, 4), y_pred_1hot)
        self.recall_tracker.update_state(ops.one_hot(y, 4), y_pred_1hot)
        return {
            "loss": v_loss,
            "acc": self.acc_tracker.result(),
            "f1": self.f1_tracker.result(),
            "precision": self.precision_tracker.result(),
            "recall": self.recall_tracker.result()
        }

    def predict_one(self, x):
        h_all = []
        for label in range(4):
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

def plot_training_results(history, output_dir):
    """Generates and saves performance plots for all tracked metrics."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    metrics = ['loss', 'acc', 'f1', 'precision', 'recall']
    
    for metric in metrics:
        if metric in history.history:
            plt.figure(figsize=(10, 6))
            plt.plot(history.history[metric], label=f'Train {metric}')
            val_metric = f'val_{metric}'
            if val_metric in history.history:
                plt.plot(history.history[val_metric], label=f'Val {metric}')
            
            plt.title(f'Model {metric.capitalize()} Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
            
            save_path = os.path.join(output_dir, f'ff_training_{metric}.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Saved {metric} plot to {save_path}")

# --- Main ---

def main():
    """Main execution block for TPU training."""
    train_path = "/kaggle/input/barbados-traffic-analysis-challenge/Train.csv"
    test_path = "/kaggle/input/barbados-traffic-analysis-challenge/TestInputSegments.csv"
    sample_sub_path = "/kaggle/input/barbados-traffic-analysis-challenge/SampleSubmission.csv"
    
    base_out = "/kaggle/working"
    analytics_dir = os.path.join(base_out, "analytics")
    submissions_dir = os.path.join(base_out, "submissions")
    os.makedirs(analytics_dir, exist_ok=True)
    os.makedirs(submissions_dir, exist_ok=True)
    
    print("Preparing Data...")
    X_train, y_train, X_val, y_val = create_dataset_splits(train_path)
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(10000).batch(64).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64)

    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weight_dict = dict(zip(classes, weights))

    global_epochs = 30
    batch_size = 64
    local_layer_epochs = 60
    total_batches = len(X_train) // batch_size
    total_train_steps = global_epochs * total_batches * local_layer_epochs
    
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0090513,
        decay_steps=total_train_steps,
        alpha=0.1
    )

    reg = keras.regularizers.L2(0.00079024)
    input_dim = X_train.shape[1]
    
    # Create model within TPU strategy scope
    with strategy.scope():
        model = FFNetwork(
            dims=[input_dim] + [64] * 10,
            kernel_regularizer=reg,
            learning_rate=lr_schedule,
            use_ema=True,
            ema_overwrite_frequency=1,
            layer_epochs=local_layer_epochs, 
            threshold=3.6,
            gamma=1.3
        ) 
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, global_clipnorm=1.0), 
            jit_compile=True
        )
    
    print(f"Training Model on TPU ({global_epochs} epochs with Cosine Decay)...")
    history = model.fit(
        train_dataset, 
        validation_data=val_dataset, 
        epochs=global_epochs, 
        class_weight=class_weight_dict,
        verbose=2
    )
    
    model_path = os.path.join(base_out, "ff_model.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    print("\nVisualizing Metrics...")
    plot_training_results(history, analytics_dir)
    
    print("\nInference on TestInputSegments (8-step horizon)...")
    congestion_map = {0: 'free flowing', 1: 'light delay', 2: 'moderate delay', 3: 'heavy delay'}
    test_df = pl.read_csv(test_path)
    
    prediction_dict = {}
    partitions = test_df.partition_by("view_label")

    for group in partitions:
        group = identify_blocks(group)
        label = group["view_label"][0]
        block_partitions = group.partition_by("block_id")

        for block in block_partitions:
            feats, _ = get_features_and_labels(block)
            
            if len(feats) < 15:
                pad_len = 15 - len(feats)
                padded_feats = jnp.concatenate([jnp.tile(feats[0], (pad_len, 1)), feats], axis=0)
                history = np.array(padded_feats).tolist()
            else:
                history = np.array(feats[-15:]).tolist()
            
            start_id = int(round(history[-1][3] * 5000))
            
            for i in range(1, 9):
                current_window = jnp.array(history[-15:]).flatten()
                input_vec = jnp.zeros(4 + len(current_window), dtype='float32')
                input_vec = input_vec.at[4:].set(current_window)
                
                p_label_idx = model.predict_one(ops.convert_to_tensor(input_vec)).numpy()
                p_label = congestion_map[p_label_idx]
                
                target_id = start_id + i
                prediction_dict[(label, target_id)] = p_label
                
                next_feat = jnp.array(history[-1])
                curr_h = next_feat[0] * 23.0
                curr_m = next_feat[1] * 59.0
                curr_m += 5
                if curr_m > 59:
                    curr_m -= 60
                    curr_h = (curr_h + 1) % 24
                
                next_feat = next_feat.at[0].set(curr_h / 23.0)
                next_feat = next_feat.at[1].set(curr_m / 59.0)
                next_feat = next_feat.at[3].set(next_feat[3] + (1/5000.0))
                history.append(np.array(next_feat).tolist())

    print("Mapping 8-step predictions to SampleSubmission template...")
    sample_sub = pl.read_csv(sample_sub_path)
    
    keys = list(prediction_dict.keys())
    values = list(prediction_dict.values())
    
    if keys:
        pred_df = pl.DataFrame({
            "view_label": [k[0] for k in keys],
            "time_segment_id": [k[1] for k in keys],
            "Predicted_Target": values
        })

        sample_sub = sample_sub.with_columns(
            pl.col("ID").str.split("_").list.get(2).cast(pl.Int32).alias("time_segment_id"),
            pl.col("ID").str.split("_").list.get(3).alias("view_label")
        )
        
        final_sub = sample_sub.join(pred_df, on=["view_label", "time_segment_id"], how="left")
        final_sub = final_sub.with_columns(
            pl.col("Predicted_Target").fill_null("free flowing").alias("Target")
        )
        final_sub = final_sub.with_columns(pl.col("Target").alias("Target_Accuracy"))
        final_select = final_sub.select(["ID", "Target", "Target_Accuracy"])
    else:
        print("Warning: No predictions generated. Filling with default.")
        final_select = sample_sub.with_columns([
            pl.lit("free flowing").alias("Target"),
            pl.lit("free flowing").alias("Target_Accuracy")
        ]).select(["ID", "Target", "Target_Accuracy"])

    output_path = os.path.join(submissions_dir, "submission.csv")
    final_select.write_csv(output_path)
    print(f"\nSubmission file generated: {output_path}")
    print(f"Total rows: {len(final_select)}")

if __name__ == "__main__":
    main()
