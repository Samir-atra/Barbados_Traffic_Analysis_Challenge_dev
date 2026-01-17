"""Module implementing the Forward-Forward algorithm with Hyperband Tuning (Kaggle TPU Version).

This script adapts the Forward-Forward (FF) neural network to use Keras Tuner's
Hyperband algorithm for hyperparameter optimization on TPU, specifically targeting
network depth and width.
"""

# Install required packages
import subprocess
import sys

print("Installing polars keras-tuner...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "keras-tuner"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "polars"])
print("Installation complete!")

import os
import matplotlib
matplotlib.use('Agg')

# Configure JAX backend for TPU
os.environ["KERAS_BACKEND"] = "jax"

import jax
import jax.numpy as jnp
from jax import random
import keras
from keras import ops
import keras_tuner as kt
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import shutil

# Set seeds for reproducibility
SEED = 42

keras.utils.set_random_seed(SEED)

# Initialize TPU
print("Initializing TPU...")
try:
    tpu = jax.devices('tpu')[0]
    print(f"TPU devices found: {jax.devices()}")
    print(f"TPU device count: {jax.device_count()}")
except:
    print("No TPU found, using CPU/GPU")

# --- Data Preparation Helpers ---

def get_features_and_labels(df):
    """Extracts and encodes features and labels from a traffic dataframe (Polars + JAX).
    
    Args:
        df: Polars DataFrame containing traffic data.
        
    Returns:
        Tuple of (features, labels) as JAX arrays.
    """
    # Ensure date/time columns are proper types
    df = df.with_columns([
        pl.col("video_time").str.to_datetime(),
        pl.col("date").str.to_datetime()
    ])

    df = df.with_columns([
        (pl.col("video_time").dt.hour() / 23.0).alias("hour"),
        (pl.col("video_time").dt.minute() / 59.0).alias("minute"),
        ((pl.col("date").dt.weekday() - 1) / 6.0).alias("day_of_week")  # Mon=1 -> 0
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
    
    congestion_map = {
        'free flowing': 0,
        'light delay': 1,
        'moderate delay': 2,
        'heavy delay': 3
    }
    df = df.with_columns(
        pl.col("congestion_enter_rating").replace(congestion_map, default=0).cast(pl.Int32).alias("enter_id")
    )
    
    # One-hot encoding for view_id
    view_ids = df["view_id"].to_numpy()
    view_ids_jax = jnp.array(view_ids)
    view_1hot = jnp.zeros((len(view_ids), 4), dtype=float)
    view_1hot = view_1hot.at[jnp.arange(len(view_ids)), view_ids_jax].set(1.0)
    
    # Signaling feature mapping
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
    
    return features, df['enter_id'].to_numpy()

def identify_blocks(group):
    """Sorts by time_segment_id and identifies continuous sequential blocks (Polars).
    
    Args:
        group: Polars DataFrame group.
        
    Returns:
        Polars DataFrame with block_id column added.
    """
    group = group.sort("time_segment_id")
    diff = group["time_segment_id"].diff().fill_null(1)
    is_break = (diff != 1).cast(pl.Int32)
    group = group.with_columns(is_break.cum_sum().alias("block_id"))
    return group

def create_dataset_splits(csv_path, val_split=0.2):
    """Processes CSV data into sequential training and validation samples.
    
    Args:
        csv_path: Path to the CSV file.
        val_split: Validation split ratio.
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val) as JAX/NumPy arrays.
    """
    df = pl.read_csv(csv_path)
    train_X, train_y = [], []
    val_X, val_y = [], []
    seq_len = 15
    
    print(f"Loading {len(df)} rows from {csv_path} with Polars...")
    
    partitions = df.partition_by("view_label")
    
    for group in partitions:
        label = group["view_label"][0]
        
        group = identify_blocks(group)
        view_X, view_y = [], []
        
        block_partitions = group.partition_by("block_id")
        
        for block in block_partitions:
            if len(block) < seq_len + 1:
                continue
            feats, labels = get_features_and_labels(block)
            
            n_samples = len(feats) - seq_len
            if n_samples > 0:
                indexer = jnp.arange(seq_len)[None, :] + jnp.arange(n_samples)[:, None]
                windows = feats[indexer].reshape(n_samples, -1)
                targets = labels[seq_len:]
                
                view_X.extend(jnp.array(windows)) 
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
        """Pads feature arrays with label slots.
        
        Args:
            X_list: List of feature arrays.
            
        Returns:
            Padded JAX array.
        """
        if not X_list:
            return jnp.zeros((0, 15*13 + 4), dtype='float32')
        X_arr = jnp.array(X_list)
        X_padded = jnp.zeros((X_arr.shape[0], 4 + X_arr.shape[1]), dtype='float32')
        X_padded = X_padded.at[:, 4:].set(X_arr)
        return X_padded

    return pad_X(train_X), jnp.array(train_y), pad_X(val_X), jnp.array(val_y)

# --- Forward-Forward Classes ---

class FFDense(keras.layers.Layer):
    """A single Forward-Forward Dense layer with local learning."""

    def __init__(self, units, num_epochs=54, kernel_regularizer=None, gamma=1.0337, 
                 threshold=1.143, learning_rate=0.001, use_ema=True, ema_overwrite_frequency=None, 
                 activation='leaky_relu', **kwargs):
        """Initializes FFDense layer.
        
        Args:
            units: Number of units in the layer.
            num_epochs: Number of local training epochs.
            kernel_regularizer: Kernel regularizer.
            gamma: Focal loss gamma parameter.
            threshold: Goodness threshold.
            learning_rate: Learning rate.
            use_ema: Whether to use exponential moving average.
            ema_overwrite_frequency: EMA overwrite frequency.
            activation: Activation function name.
            **kwargs: Additional layer arguments.
        """
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
        """Forward pass through the layer.
        
        Args:
            x: Input tensor.
            
        Returns:
            Activated output tensor.
        """
        x_norm = ops.norm(x, ord=2, axis=1, keepdims=True) + 1e-4
        h = self.dense(x / x_norm)
        return self.activation(h)

    def forward_forward(self, x_pos, x_neg, weights=None):
        """Performs forward-forward training on positive and negative samples.
        
        Args:
            x_pos: Positive samples.
            x_neg: Negative samples.
            weights: Optional sample weights.
            
        Returns:
            Tuple of (positive activations, negative activations, loss).
        """
        for i in range(self.num_epochs):
            with jax.value_and_grad(lambda params: self._compute_loss(params, x_pos, x_neg, weights)) as (loss, grads):
                # Apply gradients
                self.optimizer.apply_gradients(zip(grads, self.dense.trainable_weights))
                self.loss_metric.update_state([loss])
        
        return ops.stop_gradient(self.call(x_pos)), ops.stop_gradient(self.call(x_neg)), self.loss_metric.result()
    
    def _compute_loss(self, params, x_pos, x_neg, weights):
        """Computes the forward-forward loss.
        
        Args:
            params: Model parameters.
            x_pos: Positive samples.
            x_neg: Negative samples.
            weights: Optional sample weights.
            
        Returns:
            Scalar loss value.
        """
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
        
        return mean_loss

class MacroPrecision(keras.metrics.Metric):
    """Macro-averaged precision metric."""
    
    def __init__(self, num_classes=4, name="macro_precision", **kwargs):
        """Initializes MacroPrecision metric.
        
        Args:
            num_classes: Number of classes.
            name: Metric name.
            **kwargs: Additional metric arguments.
        """
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros")
        self.fp = self.add_weight(name="fp", shape=(num_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates metric state.
        
        Args:
            y_true: True labels (one-hot).
            y_pred: Predicted labels (one-hot).
            sample_weight: Optional sample weights.
        """
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")
        tp = ops.sum(y_true * y_pred, axis=0)
        fp = ops.sum((1 - y_true) * y_pred, axis=0)
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)

    def result(self):
        """Computes the metric result.
        
        Returns:
            Macro-averaged precision.
        """
        precisions = self.tp / (self.tp + self.fp + 1e-7)
        return ops.mean(precisions)

    def reset_state(self):
        """Resets metric state."""
        self.tp.assign(jnp.zeros(self.tp.shape, dtype=self.tp.dtype))
        self.fp.assign(jnp.zeros(self.fp.shape, dtype=self.fp.dtype))

class MacroRecall(keras.metrics.Metric):
    """Macro-averaged recall metric."""
    
    def __init__(self, num_classes=4, name="macro_recall", **kwargs):
        """Initializes MacroRecall metric.
        
        Args:
            num_classes: Number of classes.
            name: Metric name.
            **kwargs: Additional metric arguments.
        """
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros")
        self.fn = self.add_weight(name="fn", shape=(num_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates metric state.
        
        Args:
            y_true: True labels (one-hot).
            y_pred: Predicted labels (one-hot).
            sample_weight: Optional sample weights.
        """
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")
        tp = ops.sum(y_true * y_pred, axis=0)
        fn = ops.sum(y_true * (1 - y_pred), axis=0)
        self.tp.assign_add(tp)
        self.fn.assign_add(fn)

    def result(self):
        """Computes the metric result.
        
        Returns:
            Macro-averaged recall.
        """
        recalls = self.tp / (self.tp + self.fn + 1e-7)
        return ops.mean(recalls)

    def reset_state(self):
        """Resets metric state."""
        self.tp.assign(ops.zeros(self.tp.shape, dtype=self.tp.dtype))
        self.fn.assign(ops.zeros(self.fn.shape, dtype=self.fn.dtype))

class FFNetwork(keras.Model):
    """The full Forward-Forward network model."""

    def __init__(self, dims, kernel_regularizer=None, learning_rate=0.001, 
                 use_ema=True, ema_overwrite_frequency=None, 
                 layer_epochs=54, threshold=1.143, gamma=1.0337, **kwargs):
        """Initializes FFNetwork.
        
        Args:
            dims: List of layer dimensions.
            kernel_regularizer: Kernel regularizer.
            learning_rate: Learning rate.
            use_ema: Whether to use exponential moving average.
            ema_overwrite_frequency: EMA overwrite frequency.
            layer_epochs: Number of epochs per layer.
            threshold: Goodness threshold.
            gamma: Focal loss gamma parameter.
            **kwargs: Additional model arguments.
        """
        super().__init__(**kwargs)
        self.loss_var = keras.Variable(0.0, trainable=False)
        self.loss_count = keras.Variable(0.0, trainable=False)
        self.ff_layers = []
        for i, d in enumerate(dims[1:]):
            act = 'softmax' if i == len(dims[1:]) - 1 else 'leaky_relu'
            self.ff_layers.append(
                FFDense(d, kernel_regularizer=kernel_regularizer, 
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

    @property
    def metrics(self):
        """Returns list of metrics.
        
        Returns:
            List of metric objects.
        """
        return [self.acc_tracker, self.f1_tracker, self.precision_tracker, self.recall_tracker]

    def call(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        h = x
        for layer in self.ff_layers:
            h = layer(h)
        return h

    def overlay_y_on_x(self, data):
        """Overlays label onto input features.
        
        Args:
            data: Tuple of (features, label).
            
        Returns:
            Tuple of (modified features, label).
        """
        x, y = data
        x_zeros = ops.zeros([4], dtype=x.dtype)
        y_idx = ops.reshape(ops.cast(y, "int32"), [])
        
        # Create one-hot encoding for the label
        update = ops.one_hot(y_idx, 4) * 10.0
        
        # Update the first 4 positions of x with the label encoding
        x_updated = ops.concatenate([update, x[4:]], axis=0)
        return x_updated, y

    def predict_batch(self, x):
        """Predicts labels for a batch of inputs.
        
        Args:
            x: Input batch.
            
        Returns:
            Predicted labels.
        """
        return ops.vectorized_map(self.predict_one, x)

    def train_step(self, x, y, sample_weight=None):
        """Performs a single training step.
        
        Args:
            x: Input features.
            y: True labels.
            sample_weight: Optional sample weights.
            
        Returns:
            Dictionary of metric values.
        """
        if sample_weight is not None:
            weights = ops.cast(sample_weight, "float32")
                # Ensure y has at least one dimension for ops.vectorized_map
                if ops.ndim(y) == 0:
                    y = ops.expand_dims(y, axis=0)
                x_pos, _ = ops.vectorized_map(self.overlay_y_on_x, (x, y))
        
        batch_size = ops.shape(y)[0]
        # Generate random offsets for negative samples
        rng = random.PRNGKey(SEED)
        offsets = random.randint(rng, shape=[batch_size], minval=1, maxval=4)
        y_neg_labels = (ops.cast(y, "int32") + offsets) % 4
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

    def test_step(self, x, y, sample_weight=None):
        """Performs a single validation/test step.
        
        Args:
            x: Input features.
            y: True labels.
            sample_weight: Optional sample weights.
            
        Returns:
            Dictionary of metric values.
        """
        if sample_weight is not None:
            weights = ops.cast(sample_weight, "float32")
        else:
            weights = None
        
        x_pos, _ = ops.vectorized_map(self.overlay_y_on_x, (x, y))
        
        batch_size = ops.shape(y)[0]
        rng = random.PRNGKey(SEED)
        offsets = random.randint(rng, shape=[batch_size], minval=1, maxval=4)
        y_neg_labels = (ops.cast(y, "int32") + offsets) % 4
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
        """Predicts label for a single input.
        
        Args:
            x: Input tensor.
            
        Returns:
            Predicted label.
        """
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

# --- Hyperband Model ---

class FFHyperModel(kt.HyperModel):
    """Hyperband model for FF network hyperparameter tuning."""
    
    def __init__(self, input_dim, total_train_steps):
        """Initializes FFHyperModel.
        
        Args:
            input_dim: Input dimension.
            total_train_steps: Total number of training steps.
        """
        self.input_dim = input_dim
        self.total_train_steps = total_train_steps

    def build(self, hp):
        """Builds a model with hyperparameters.
        
        Args:
            hp: HyperParameters object.
            
        Returns:
            Compiled FFNetwork model.
        """
        # Tune the number of layers (4-16)
        num_layers = hp.Int('num_layers', 4, 16)
        
        # Tune the units per layer (16-256)
        units = hp.Int('units', 16, 256, step=16)
        
        # Tune Learning Rate
        lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        
        # Tune FF-specific hyperparameters
        layer_epochs = hp.Int('layer_epochs', 20, 80, step=5)
        threshold = hp.Float('threshold', 0.5, 4.0, step=0.1)
        gamma = hp.Float('gamma', 1.0, 5.0, step=0.1)
        
        # Tune Regularization
        l2_reg = hp.Float('l2_reg', 1e-6, 1e-3, sampling='log')
        
        # Tune EMA frequency
        ema_overwrite_frequency = hp.Int('ema_overwrite_frequency', 1, 10)
        
        # Construct dimensions: Input -> Hidden Layers -> Output
        dims = [self.input_dim] + [units] * num_layers
        
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=self.total_train_steps,
            alpha=0.1
        )
        
        model = FFNetwork(
            dims=dims,
            kernel_regularizer=keras.regularizers.L2(l2_reg),
            learning_rate=lr_schedule,
            use_ema=True,
            ema_overwrite_frequency=ema_overwrite_frequency,
            layer_epochs=layer_epochs,
            threshold=threshold,
            gamma=gamma
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, global_clipnorm=1.0),
            jit_compile=False  # Disable XLA for TPU compatibility
        )
        return model

# --- Main ---

def main():
    """Main execution block for Hyperband Tuning on TPU."""
    # Kaggle specific paths
    train_path = "/kaggle/input/barbados-traffic-analysis-challenge/Train.csv"
    
    # Output directories in Kaggle working directory
    base_out = "/kaggle/working"
    tuner_dir = os.path.join(base_out, "kt_hyperband")
    
    # Clean previous tuner results if needed to force fresh search
    # if os.path.exists(tuner_dir):
    #     shutil.rmtree(tuner_dir)
    
    print("Preparing Data for Hyperparameter Search...")
    X_train, y_train, X_val, y_val = create_dataset_splits(train_path)
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    # Convert to NumPy for compatibility with Keras and Scikit-learn
    X_train_np = jnp.array(X_train)
    y_train_np = np.array(y_train)
    X_val_np = jnp.array(X_val)
    y_val_np = np.array(y_val)
    
    # Compute class weights
    classes = np.array(jnp.unique(y_train_np))
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train_np
    )
    class_weight_dict = dict(zip(classes, weights))
    
    print(f"Class weights: {class_weight_dict}")

    # Hyperparameter search configuration
    global_epochs = 10  # Reduced for TPU tuning speed
    batch_size = 128  # Larger batch size for TPU efficiency
    local_layer_epochs = 43
    total_batches = len(X_train_np) // batch_size
    total_train_steps = global_epochs * total_batches * local_layer_epochs
    
    input_dim = X_train_np.shape[1]
    
    hypermodel = FFHyperModel(input_dim, total_train_steps)
    
    tuner = kt.Hyperband(
        hypermodel,
        objective=kt.Objective("val_acc", direction="max"),
        max_epochs=global_epochs,
        factor=3,
        seed=SEED,
        directory=base_out,
        project_name='ff_traffic_hyperband_tpu',
    )
    
    print("Starting Hyperband Search on TPU...")
    print(f"Max epochs: {global_epochs}, Batch size: {batch_size}")
    
    tuner.search(
        X_train_np,
        y_train_np,
        validation_data=(X_val_np, y_val_np),
        epochs=global_epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print(f"""
    ========================================
    Hyperparameter Search Complete!
    ========================================
    Optimal number of layers: {best_hps.get('num_layers')}
    Optimal number of units: {best_hps.get('units')}
    Optimal learning rate: {best_hps.get('learning_rate'):.6f}
    Optimal layer epochs: {best_hps.get('layer_epochs')}
    Optimal threshold: {best_hps.get('threshold'):.3f}
    Optimal gamma: {best_hps.get('gamma'):.3f}
    Optimal L2 regularization: {best_hps.get('l2_reg'):.6f}
    Optimal EMA frequency: {best_hps.get('ema_overwrite_frequency')}
    ========================================
    """)
    
    print("Retraining best model with extended epochs...")
    best_model = tuner.hypermodel.build(best_hps)
    history = best_model.fit(
        X_train_np,
        y_train_np,
        validation_data=(X_val_np, y_val_np),
        epochs=global_epochs + 5,  # Train a bit longer
        batch_size=batch_size,
        class_weight=class_weight_dict,
        verbose=2
    )
    
    # Save best model weights
    weights_path = os.path.join(base_out, "best_ff_tpu_weights.weights.h5")
    best_model.save_weights(weights_path)
    print(f"Best model weights saved to: {weights_path}")
    
    # Save hyperparameters to file
    hp_path = os.path.join(base_out, "best_hyperparameters.txt")
    with open(hp_path, 'w') as f:
        f.write("Best Hyperparameters:\n")
        f.write("=" * 50 + "\n")
        for key in best_hps.values.keys():
            f.write(f"{key}: {best_hps.get(key)}\n")
    print(f"Hyperparameters saved to: {hp_path}")

if __name__ == "__main__":
    main()
