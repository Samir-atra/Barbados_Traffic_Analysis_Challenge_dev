"""
Module to perform hyperparameter tuning for the Forward-Forward algorithm using Hyperband."""

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
from sklearn.utils.class_weight import compute_class_weight

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
keras.utils.set_random_seed(SEED)

# --- Data Preparation Helpers ---

def get_features_and_labels(df):
    """Extracts and encodes features and labels from a traffic dataframe.
    
    This function performs feature engineering on temporal, spatial, and 
    signaling data. It maps categorical congestion ratings to integers and 
    one-hot encodes view labels and signaling states.

    Args:
        df: A pandas DataFrame containing raw traffic data.

    Returns:
        A tuple (features, labels):
            - features: A float32 NumPy array of shape (N, 13).
            - labels: An int32 NumPy array of shape (N,) containing joint labels.
    """
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
    
    # signaling feature mapping
    sig_map = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
    df['sig_id'] = df['signaling'].map(sig_map).fillna(0)
    sig_1hot = pd.get_dummies(df['sig_id'], prefix='sig').reindex(
        columns=['sig_0', 'sig_1', 'sig_2', 'sig_3'], fill_value=0).astype(float).values
    
    features = np.concatenate([
        df[['hour', 'minute', 'day_of_week', 'seg_id_norm', 'view_id']].values,
        view_1hot,
        sig_1hot
    ], axis=1).astype('float32') # 5 base + 4 view-1hot + 4 sig-1hot = 13 features
    
    return features, df['enter_id'].values

def identify_blocks(group):
    """Sorts by time_segment_id and identifies continuous sequential blocks."""
    group = group.sort_values('time_segment_id')
    ids = group['time_segment_id'].values
    is_break = np.zeros(len(ids), dtype=int)
    is_break[1:] = (ids[1:] != ids[:-1] + 1).astype(int)
    group['block_id'] = np.cumsum(is_break)
    return group

def create_dataset_splits(csv_path, val_split=0.2):
    """Processes CSV data into sequential training and validation samples.
    
    Groups traffic records by view_label and sequential time blocks. For each
    block, it creates input/target pairs where input is the current state
    and target is the congestion level at the next time step. Splits are done
    per view to ensure all views are represented in both sets.

    Args:
        csv_path: Path to the Train.csv file.
        val_split: Fraction of data to use for validation (default 0.2).

    Returns:
        A tuple (X_train, y_train, X_val, y_val).
    """
    df = pd.read_csv(csv_path)
    train_X, train_y = [], []
    val_X, val_y = [], []
    seq_len = 15
    
    print(f"Loading {len(df)} rows from {csv_path}...")
    for label, group in df.groupby('view_label'):
        # print(f"  Processing view: {label}")
        group = identify_blocks(group)
        view_X, view_y = [], []
        
        for b_id, block in group.groupby('block_id'):
            if len(block) < seq_len + 1: continue
            feats, labels = get_features_and_labels(block)
            for i in range(len(feats) - seq_len):
                # Concatenate 15 steps of features: [Xt-14, ..., Xt]
                window = feats[i : i + seq_len].flatten()
                view_X.append(window)
                view_y.append(labels[i + seq_len])
        
        # Split this view's data
        n_total = len(view_X)
        if n_total > 0:
            n_val = int(n_total * val_split)
            n_train = n_total - n_val
            
            train_X.extend(view_X[:n_train])
            train_y.extend(view_y[:n_train])
            val_X.extend(view_X[n_train:])
            val_y.extend(view_y[n_train:])
                
    # Convert to arrays and pad with label buffer
    def pad_X(X_list):
        X_arr = np.array(X_list)
        # Pad with 4 zeros for 4 classes
        X_padded = np.zeros((X_arr.shape[0], 4 + X_arr.shape[1]), dtype='float32')
        X_padded[:, 4:] = X_arr
        return X_padded

    return pad_X(train_X), np.array(train_y), pad_X(val_X), np.array(val_y)

# --- Forward-Forward Classes ---

class FFDense(keras.layers.Layer):
    """A single Forward-Forward Dense layer with local learning."""
    def __init__(self, units, num_epochs=54, kernel_regularizer=None, gamma=1.0337, 
                 threshold=1.143, learning_rate=0.001, use_ema=True, ema_overwrite_frequency=None, 
                 activation='leaky_relu', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = keras.layers.Dense(
            units=units, 
            kernel_regularizer=kernel_regularizer,
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
        self.threshold = threshold
        self.num_epochs = num_epochs
        self.gamma = gamma

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

class MacroPrecision(keras.metrics.Metric):
    """Computes Macro-Averaged Precision for multi-class classification."""
    def __init__(self, num_classes=4, name="macro_precision", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros")
        self.fp = self.add_weight(name="fp", shape=(num_classes,), initializer="zeros")

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

class MacroRecall(keras.metrics.Metric):
    """Computes Macro-Averaged Recall for multi-class classification."""
    def __init__(self, num_classes=4, name="macro_recall", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros")
        self.fn = self.add_weight(name="fn", shape=(num_classes,), initializer="zeros")

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

class FFNetwork(keras.Model):
    def __init__(self, dims, layer_epochs=54, threshold=1.143, learning_rate=0.001, 
                 kernel_regularizer=None, gamma=1.0337, use_ema=True, ema_overwrite_frequency=None, **kwargs):
        super().__init__(**kwargs)
        self.loss_var = keras.Variable(0.0, trainable=False)
        self.loss_count = keras.Variable(0.0, trainable=False)
        self.ff_layers = []
        for i, d in enumerate(dims[1:]):
            # Use softmax for the last layer, LeakyReLU for others
            act = 'softmax' if i == len(dims[1:]) - 1 else 'leaky_relu'
            self.ff_layers.append(
                FFDense(d, num_epochs=layer_epochs, threshold=threshold, learning_rate=learning_rate, 
                        kernel_regularizer=kernel_regularizer, gamma=gamma, 
                        use_ema=use_ema, ema_overwrite_frequency=ema_overwrite_frequency,
                        activation=act)
            )
        self.acc_tracker = keras.metrics.SparseCategoricalAccuracy(name="acc")
        self.f1_tracker = keras.metrics.F1Score(name="f1", average="macro")
        self.precision_tracker = MacroPrecision(name="precision")
        self.recall_tracker = MacroRecall(name="recall")
        self.build((None, dims[0]))

    @property
    def metrics(self):
        return [self.acc_tracker, self.f1_tracker, self.precision_tracker, self.recall_tracker]

    def overlay_y_on_x(self, data):
        x, y = data
        x_zeros = ops.zeros([4], dtype=x.dtype)
        y_idx = ops.reshape(ops.cast(y, "int32"), [])
        update = xla.dynamic_update_slice(x_zeros, [ops.cast(10.0, x.dtype)], [y_idx])
        return xla.dynamic_update_slice(x, update, [0]), y

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
        return {"loss": v_loss, "acc": self.acc_tracker.result(), "f1": self.f1_tracker.result(), "precision": self.precision_tracker.result(), "recall": self.recall_tracker.result()}

    def predict_one(self, x):
        h_all = []
        for label in range(4):
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

def build_model(hp, input_dim=None, train_size=None):
    """Builds a model with hyperparameters from the search space."""
    if input_dim is None:
        # Default to 17 (13 features + 4 label padding)
        input_dim = 17
    
    # Architecture search: Try both uniform blocks and tapering
    # For simplicity in Hyperband, we'll stick to uniform or simple variations
    # Restructured used: [128, 128, 128, 128, 128, 64, 64, 64, 32, 16]
    
    num_layers = hp.Int("num_layers", 2, 12)
    units = hp.Choice("units", [64, 128, 256])
    
    # Restructured best: LR ~ 7.4e-4
    # Base LR for the schedule
    initial_learning_rate = hp.Float("learning_rate", 1e-4, 5e-3, sampling="log")
    
    # Restructured best: Threshold ~ 1.04
    threshold = hp.Float("threshold", 0.8, 2.0)
    
    # Restructured best: Epochs ~ 43
    layer_epochs = hp.Int("layer_epochs", 20, 80)
    
    # Restructured best: Reg ~ 1.5e-5
    l2_reg = hp.Float("l2_reg", 1e-6, 1e-3, sampling="log")
    
    # Restructured best: Gamma ~ 2.48
    gamma = hp.Float("gamma", 1.0, 4.0)
    
    use_ema = hp.Boolean("use_ema", default=True)
    ema_overwrite_frequency = hp.Int("ema_overwrite_frequency", 1, 10) if use_ema else 1
    taper = hp.Boolean("taper_layers", default=True)
    
    # Learning Rate Schedule
    # If train_size is provided, use CosineDecay targeting 30 global epochs
    if train_size is not None:
        batch_size = 64
        global_epochs = 30
        total_batches = train_size // batch_size
        # Total steps for the layer-wise optimizer
        total_train_steps = global_epochs * total_batches * layer_epochs
        
        learning_rate = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=total_train_steps,
            alpha=0.1
        )
    else:
        learning_rate = initial_learning_rate
    
    dims = [input_dim]
    current_units = units
    for i in range(num_layers):
        dims.append(current_units)
        if taper and i > 0 and i % 3 == 0:
            current_units = max(16, current_units // 2)
            
    reg = keras.regularizers.L2(l2_reg)
    
    model = FFNetwork(
        dims=dims,
        layer_epochs=layer_epochs,
        threshold=threshold,
        learning_rate=learning_rate,
        kernel_regularizer=reg,
        gamma=gamma,
        use_ema=use_ema,
        ema_overwrite_frequency=ema_overwrite_frequency,
    )
    
    # Compile with gradient clipping for stability
    # Using 'f1' (macro F1) as the key metric
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, global_clipnorm=1.0),
        jit_compile=False,
        metrics=["acc", "f1", "precision", "recall"]
    )
    return model

def main():
    base = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev"
    train_path = os.path.join(base, "demos/Train.csv")
    
    print("Loading and preparing data...")
    # X_train, y_train, X_val, y_val are numpy arrays
    X_train, y_train, X_val, y_val = create_dataset_splits(train_path)
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"Input dimension: {X_train.shape[1]}, Num classes: {len(np.unique(y_train))}")
    
    batch_size = 64
    
    # Compute class weights for balanced training (4 classes)
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weight_dict = dict(zip(classes, weights))
    print(f"Class weights computed: {class_weight_dict}")
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

    # Create a wrapper function that passes input_dim
    input_dim = X_train.shape[1]
    train_size = len(X_train)
    def build_model_wrapper(hp):
        return build_model(hp, input_dim=input_dim, train_size=train_size)
    
    tuner = kt.RandomSearch(
        build_model_wrapper,
        objective=kt.Objective("val_f1", direction="max"),
        # max_epochs=30,
        # factor=3,
        max_trials=10,
        executions_per_trial=2,
        directory=os.path.join(base, "hyperband_search"),
        project_name="ff_traffic_4class_v2", # Changed project name to avoid conflict/confusion with old 16-class
        seed=SEED,
        overwrite=True
    )

    print("\nStarting hyperparameter search...")
    tuner.search(
        train_dataset,
        validation_data=val_dataset,
        class_weight=class_weight_dict,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_f1',
                patience=5,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            )
        ],
        verbose=2
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\n" + "="*60)
    print("Best Hyperparameters Found:")
    print("="*60)
    print(f"Num Layers: {best_hps.get('num_layers')}")
    print(f"Initial Units: {best_hps.get('units')}")
    print(f"Taper Layers: {best_hps.get('taper_layers')}")
    print(f"Learning Rate: {best_hps.get('learning_rate'):.6f}")
    print(f"Threshold: {best_hps.get('threshold'):.4f}")
    print(f"Layer Epochs: {best_hps.get('layer_epochs')}")
    print(f"L2 Reg: {best_hps.get('l2_reg'):.8f}")
    print(f"Gamma: {best_hps.get('gamma'):.4f}")
    if best_hps.get('use_ema'):
        print(f"EMA Overwrite Frequency: {best_hps.get('ema_overwrite_frequency')}")
    print("="*60)

if __name__ == "__main__":
    main()