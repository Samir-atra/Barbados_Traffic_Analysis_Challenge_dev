"""Module implementing the Forward-Forward algorithm for traffic prediction.

This script implements a Forward-Forward (FF) neural network to predict future
traffic congestion states based on sequential historical data. It handles
dataset imbalance using weighted sampling and includes visualization for
model evaluation metrics.

Workflow:
1. Load and preprocess traffic data from Train.csv.
2. Group data into sequential blocks for time-series prediction.
3. Train a Forward-Forward network with Focal Loss and Layer Normalization.
4. Address class imbalance through balanced class sampling.
5. Predict the next 5 congestion states for TestInputSegments.csv.
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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.compiler.tf2xla.python import xla
import random

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
keras.utils.set_random_seed(SEED)

# Global training flag
IS_TRAINING = True

# --- Global Model Parameters ---
HIDDEN_ACTIVATION = 'leaky_relu'
LAST_LAYER_ACTIVATION = 'leaky_relu'

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
    # print(features, len(df))
    return features#, df['enter_id'].values

def identify_blocks(group):
    """Sorts by time_segment_id and identifies continuous sequential blocks."""
    group = group.sort_values('time_segment_id')
    ids = group['time_segment_id'].values
    is_break = np.zeros(len(ids), dtype=int)
    is_break[1:] = (ids[1:] != ids[:-1] + 1).astype(int)
    group['block_id'] = np.cumsum(is_break)
    return group

def create_dataset_splits(csv_path, val_split):
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
        print(f"  Processing view: {label}")
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
        X_padded = np.zeros((X_arr.shape[0], 4 + X_arr.shape[1]), dtype='float32')
        X_padded[:, 4:] = X_arr
        return X_padded

    return pad_X(train_X), np.array(train_y), pad_X(val_X), np.array(val_y)

def categorical_focal_loss(y_true, y_pred, gamma):
    """Computes Categorical Focal Loss using Keras ops.
    
    Args:
        y_true: One-hot encoded ground truth labels.
        y_pred: Softmax probabilities from the model.
        gamma: Focusing parameter.
        
    Returns:
        Loss tensor.
    """
    y_pred = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
    focal_weight = ops.power(1.0 - y_pred, gamma)
    return -ops.sum(y_true * focal_weight * ops.log(y_pred), axis=-1)

# --- Forward-Forward Classes ---

@keras.saving.register_keras_serializable(package="MyLayers")
class FFDense(keras.layers.Layer):
    """A single Forward-Forward Dense layer with local categorical focal learning.
    
    This layer implements the core FF logic: it trains itself using local 
    goodness scores without global backpropagation. It includes LayerNorm 
    and LeakyReLU for stability and employs Categorical Focal Loss.
    """

    def __init__(self, units, num_epochs, patience, min_delta, dropout_rate, kernel_regularizer, gamma, 
                 threshold, learning_rate, use_ema, ema_overwrite_frequency, 
                 activation, **kwargs):
        """Initializes the FFDense layer.

        Args:
            units: Number of hidden units.
            num_epochs: Local training epochs per global epoch.
            patience: Number of epochs with no improvement after which training will be stopped.
            min_delta: Minimum change in the monitored loss to qualify as an improvement.
            dropout_rate: Fraction of the input units to drop.
            kernel_regularizer: Keras regularizer for the dense weights.
            gamma: Focusing parameter for Focal Loss.
            learning_rate: Learning rate for the local optimizer.
            use_ema: Whether to use Exponential Moving Average.
            activation: Activation function to use ('leaky_relu', 'softmax', etc.).
            **kwargs: Standard layer arguments.
        """
        super().__init__(**kwargs)
        self.units = units
        self.num_epochs = num_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.dropout_rate = dropout_rate
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.gamma = gamma
        self.threshold = threshold
        
        # Handle learning_rate if it's a serialized dict
        if isinstance(learning_rate, dict):
            learning_rate = keras.saving.deserialize_keras_object(learning_rate)
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
        
        self.dropout = keras.layers.Dropout(dropout_rate)
            
        # Apply EMA and Clipping directly to the layer-wise optimizer
        self.optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate, 
            global_clipnorm=1.0, 
            use_ema=use_ema,
            ema_overwrite_frequency=ema_overwrite_frequency
        )
        self.loss_metric = keras.metrics.Mean()
        
        # Initialize early stopping variables as tf.Variables
        self.best_loss_var = tf.Variable(float('inf'), dtype=tf.float32, trainable=False)
        self.wait_var = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.continue_training_flag = tf.Variable(True, dtype=tf.bool, trainable=False)


    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "num_epochs": self.num_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "dropout_rate": self.dropout_rate,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "gamma": self.gamma,
            "threshold": self.threshold,
            "learning_rate": self.learning_rate,
            "use_ema": self.use_ema,
            "ema_overwrite_frequency": self.ema_overwrite_frequency,
            "activation": self.activation_name,
        })
        return config

    def build(self, input_shape):
        self.dense.build(input_shape)
        super().build(input_shape)

    def call(self, x, training=None):
        """Forward pass of the layer without normalization.

        Args:
            x: Input tensor.
            training: Boolean, whether the call is in training mode.

        Returns:
            Activated layer output.
        """
        if training is None:
            training = IS_TRAINING
        h = self.dense(x)
        h = self.activation(h)
        if training:
            h = self.dropout(h, training=training)
        return h

    def forward_forward(self, x_all, y_true, weights):
        """Local training logic using Categorical Focal Loss.

        Updates layer weights to maximize 'goodness' for the correct label 
        relative to incorrect labels using Categorical Focal Loss.

        Args:
            x_all: Input with all 4 classes overlaid (batch, 4, dim).
            y_true: True integer labels (batch,).
            weights: Optional sample weights.

        Returns:
            A tuple (h_all, loss):
                - h_all: Layer activations for all 4 class-overlaid versions.
                - loss: Mean categorical focal loss for this layer.
        """
        y_true_1hot = ops.one_hot(y_true, 4)
        
        # Reset early stopping variables for the current local training session
        self.best_loss_var.assign(float('inf'))
        self.wait_var.assign(0)
        self.continue_training_flag.assign(True)

        i = tf.constant(0)

        def loop_cond(i, best_loss, wait, continue_training):
            return tf.logical_and(tf.less(i, self.num_epochs), continue_training)

        def loop_body(i, current_best_loss, current_wait, continue_training_flag):
            with tf.GradientTape() as tape:
                h_all_curr = self.call(x_all, training=True)
                g_all = ops.mean(ops.power(h_all_curr, 2), axis=-1)
                probs = ops.softmax(g_all, axis=-1)
                
                loss = categorical_focal_loss(y_true_1hot, probs, gamma=self.gamma)
                
                if weights is not None:
                    loss = loss * weights
                
                mean_loss = ops.cast(ops.mean(loss), dtype="float32")
                if self.dense.losses:
                    mean_loss += ops.sum(self.dense.losses)
                self.loss_metric.update_state([mean_loss])
                
            grads = tape.gradient(mean_loss, self.dense.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.dense.trainable_weights))

            # Early stopping check: return new tensor values
            is_improvement = tf.less(mean_loss, current_best_loss - self.min_delta)
            
            new_best_loss = tf.cond(is_improvement, lambda: mean_loss, lambda: current_best_loss)
            new_wait = tf.cond(is_improvement, lambda: tf.constant(0, dtype=tf.int32), lambda: current_wait + 1)
            
            new_continue_training_flag = tf.less(new_wait, self.patience)
            
            return i + 1, new_best_loss, new_wait, new_continue_training_flag

        # Execute the while_loop
        final_i, final_best_loss, final_wait, final_continue_training_flag = tf.while_loop(
            loop_cond, loop_body, 
            [i, self.best_loss_var.read_value(), self.wait_var.read_value(), self.continue_training_flag.read_value()],
            maximum_iterations=self.num_epochs
        )

        # Assign the final values back to the layer's tf.Variables
        self.best_loss_var.assign(final_best_loss)
        self.wait_var.assign(final_wait)
        self.continue_training_flag.assign(final_continue_training_flag)
        
        return ops.stop_gradient(self.call(x_all, training=True)), self.loss_metric.result()

@keras.saving.register_keras_serializable(package="MyMetrics")
class MacroPrecision(keras.metrics.Metric):
    """Computes Macro-Averaged Precision for multi-class classification."""
    def __init__(self, num_classes=4, name="macro_precision", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros")
        self.fp = self.add_weight(name="fp", shape=(num_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight):
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")
        
        # Calculate per-class True Positives and False Positives
        tp = ops.sum(y_true * y_pred, axis=0)
        fp = ops.sum((1 - y_true) * y_pred, axis=0)
        
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)

    def result(self):
        # Precision = TP / (TP + FP)
        # Add epsilon to avoid division by zero
        precisions = self.tp / (self.tp + self.fp + 1e-7)
        return ops.mean(precisions)

    def reset_state(self):
        self.tp.assign(ops.zeros_like(self.tp))
        self.fp.assign(ops.zeros_like(self.fp))
        
    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config

@keras.saving.register_keras_serializable(package="MyMetrics")
class MacroRecall(keras.metrics.Metric):
    """Computes Macro-Averaged Recall for multi-class classification."""
    def __init__(self, num_classes=4, name="macro_recall", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros")
        self.fn = self.add_weight(name="fn", shape=(num_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight):
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")
        
        # Calculate per-class True Positives and False Negatives
        tp = ops.sum(y_true * y_pred, axis=0)
        fn = ops.sum(y_true * (1 - y_pred), axis=0)
        
        self.tp.assign_add(tp)
        self.fn.assign_add(fn)

    def result(self):
        # Recall = TP / (TP + FN)
        recalls = self.tp / (self.tp + self.fn + 1e-7)
        return ops.mean(recalls)

    def reset_state(self):
        self.tp.assign(ops.zeros_like(self.tp))
        self.fn.assign(ops.zeros_like(self.fn))
        
    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config

@keras.saving.register_keras_serializable(package="MyModels")
class FFNetwork(keras.Model):
    """The full Forward-Forward network model with Categorical Focal objective.

    Coordinates layer-wise categorical training and prediction.
    """

    def __init__(self, dims, kernel_regularizer, learning_rate, 
                 use_ema, ema_overwrite_frequency, 
                 layer_epochs, threshold, gamma,
                 dropout_rate, patience, min_delta,
                 hidden_activation=HIDDEN_ACTIVATION,
                 last_layer_activation=LAST_LAYER_ACTIVATION, **kwargs):
        """Initializes the network.

        Args:
            dims: List of dimensions [input_dim, layer1_units, ...].
            kernel_regularizer: Regularizer for all dense layers.
            learning_rate: LR passed to FF layers.
            use_ema: EMA flag passed to FF layers.
            layer_epochs: Training epochs for each layer.
            threshold: Goodness threshold.
            gamma: Focal loss focusing parameter.
            dropout_rate: Dropout rate for hidden layers.
            patience: Early stopping patience for local training.
            min_delta: Early stopping min_delta for local training.
            hidden_activation: Activation for hidden layers.
            last_layer_activation: Activation for the last layer.
            **kwargs: Standard model arguments.
        """
        super().__init__(**kwargs)
        self.dims = dims
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.learning_rate = learning_rate
        self.use_ema = use_ema
        self.ema_overwrite_frequency = ema_overwrite_frequency
        self.layer_epochs = layer_epochs
        self.threshold = threshold
        self.gamma = gamma
        self.dropout_rate = dropout_rate
        self.patience = patience
        self.min_delta = min_delta
        self.hidden_activation = hidden_activation
        self.last_layer_activation = last_layer_activation
        
        self.loss_var = keras.Variable(0.0, trainable=False)
        self.loss_count = keras.Variable(0.0, trainable=False)
        self.ff_layers = []
        for i, d in enumerate(dims[1:]):
            # Use softmax for the last layer, LeakyReLU for others
            act = self.last_layer_activation if i == len(dims[1:]) - 1 else self.hidden_activation
            self.ff_layers.append(
                FFDense(d, kernel_regularizer=self.kernel_regularizer, 
                        learning_rate=learning_rate, use_ema=use_ema,
                        ema_overwrite_frequency=ema_overwrite_frequency,
                        num_epochs=layer_epochs, threshold=threshold, gamma=gamma,
                        dropout_rate=dropout_rate, patience=patience, 
                        min_delta=min_delta, activation=act)
            )
        self.acc_tracker = keras.metrics.SparseCategoricalAccuracy(name="acc")
        self.f1_tracker = keras.metrics.F1Score(name="f1", average="macro")
        self.precision_tracker = MacroPrecision(name="precision")
        self.recall_tracker = MacroRecall(name="recall")
        self.focal_tracker = keras.metrics.Mean(name="focal")
        # Explicitly build the model to allow weight saving
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
            "dropout_rate": self.dropout_rate,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "hidden_activation": self.hidden_activation,
            "last_layer_activation": self.last_layer_activation,
        })
        return config

    def build(self, input_shape):
        current_shape = input_shape
        for layer in self.ff_layers:
            layer.build(current_shape)
            current_shape = (current_shape[0], layer.units)
        super().build(input_shape)

    @property
    def metrics(self):
        """Returns the model's metrics for tracking."""
        return [self.acc_tracker, self.f1_tracker, self.precision_tracker, self.recall_tracker, self.focal_tracker]

    def call(self, x, training):
        """Standard forward pass through all layers for Keras tracing."""
        if training is None:
            training = IS_TRAINING
        h = x
        for layer in self.ff_layers:
            h = layer(h, training=training)
        return h

    def overlay_y_on_x(self, data):
        """Overlays the label 'y' onto the first 4 dimensions of input 'x'.
        
        Args:
            data: Tuple of (x, y).

        Returns:
            Input vector with one-hot label encoded in the header.
        """
        x, y = data
        x_zeros = ops.zeros([4], dtype=x.dtype)
        # Ensure y is a scalar for indexing
        y_idx = ops.reshape(ops.cast(y, "int32"), [])
        update = xla.dynamic_update_slice(x_zeros, [ops.cast(10.0, x.dtype)], [y_idx])
        return xla.dynamic_update_slice(x, update, [0]), y

    def overlay_all_labels(self, x):
        """Creates 4 versions of the input, each with a different class label overlaid.
        
        Args:
            x: Input batch (batch, dim).
            
        Returns:
            Tensor of shape (batch, 4, dim).
        """
        def get_all(single_x):
            res = []
            for i in range(4):
                res.append(self.overlay_y_on_x((single_x, i))[0])
            return ops.stack(res)
        return ops.vectorized_map(get_all, x)

    @tf.function
    def predict_batch(self, x):
        """Predicts labels for a batch of inputs using vectorized_map."""
        return ops.vectorized_map(self.predict_one, x)

    def train_step(self, data):
        """Custom training step implementing layer-wise categorical updates."""
        if len(data) == 3:
            x, y, weights = data
            weights = ops.cast(weights, "float32")
        else:
            x, y = data
            weights = None

        x_all = self.overlay_all_labels(x)
        y_true_1hot = ops.one_hot(y, 4)
        
        self.loss_var.assign(0.0)
        self.loss_count.assign(0.0)
        
        h_all = x_all
        total_goodness = ops.zeros((ops.shape(x)[0], 4), dtype="float32")
        
        for layer in self.ff_layers:
            h_all, loss = layer.forward_forward(h_all, y, weights=weights)
            self.loss_var.assign_add(loss)
            self.loss_count.assign_add(1.0)
            
            # Accumulate goodness across layers for the global metric
            layer_goodness = ops.mean(ops.power(h_all, 2), axis=-1)
            total_goodness += layer_goodness
        
        # Calculate global Categorical Focal Loss for monitoring
        probs = ops.softmax(total_goodness, axis=-1)
        global_focal_loss = ops.mean(categorical_focal_loss(y_true_1hot, probs, gamma=self.gamma))
        self.focal_tracker.update_state(global_focal_loss)
        
        # Update training metrics
        y_pred = ops.argmax(total_goodness, axis=-1)
        y_pred_1hot = ops.one_hot(y_pred, 4)
        self.acc_tracker.update_state(y, y_pred_1hot)
        self.f1_tracker.update_state(y_true_1hot, y_pred_1hot)
        self.precision_tracker.update_state(y_true_1hot, y_pred_1hot, sample_weight=None)
        self.recall_tracker.update_state(y_true_1hot, y_pred_1hot, sample_weight=None)
        
        return {
            "loss": global_focal_loss, 
            "layer_loss": self.loss_var / self.loss_count,
            "acc": self.acc_tracker.result(),
            "f1": self.f1_tracker.result(),
            "focal": self.focal_tracker.result()
        }

    def test_step(self, data):
        """Custom test step using Categorical Focal objective."""
        if len(data) == 3:
            x, y, weights = data
            weights = ops.cast(weights, "float32")
        else:
            x, y = data
            weights = None
        
        x_all = self.overlay_all_labels(x)
        y_true_1hot = ops.one_hot(y, 4)
        
        total_goodness = ops.zeros((ops.shape(x)[0], 4), dtype="float32")
        h_all = x_all
        for layer in self.ff_layers:
            h_all = layer(h_all)
            layer_goodness = ops.mean(ops.power(h_all, 2), axis=-1)
            total_goodness += layer_goodness
        
        probs = ops.softmax(total_goodness, axis=-1)
        global_focal_loss = ops.mean(categorical_focal_loss(y_true_1hot, probs, gamma=self.gamma))
        self.focal_tracker.update_state(global_focal_loss)

        y_pred = ops.argmax(total_goodness, axis=-1)
        y_pred_1hot = ops.one_hot(y_pred, 4)
        self.acc_tracker.update_state(y, y_pred_1hot)
        self.f1_tracker.update_state(y_true_1hot, y_pred_1hot)
        self.precision_tracker.update_state(y_true_1hot, y_pred_1hot, sample_weight=None)
        self.recall_tracker.update_state(y_true_1hot, y_pred_1hot, sample_weight=None)
        
        return {
            "loss": global_focal_loss,
            "acc": self.acc_tracker.result(),
            "f1": self.f1_tracker.result(),
            "focal": self.focal_tracker.result()
        }

    def predict_one(self, x):
        """Predicts the label for a single sample by comparing goodness scores.

        For each possible label (0-3), it calculates the sum of goodness 
        (mean squared activation) across all layers. The label with the highest
        total goodness is selected.

        Args:
            x: Raw input vector (padded with 4 zeros).

        Returns:
            Predicted enter rating index.
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

def plot_training_results(history, output_dir):
    """Generates and saves performance plots for all tracked metrics.

    Args:
        history: Keras history object returned by model.fit().
        output_dir: Path to directory where PNG files will be saved.
    """
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
    """Main execution block for training and inference.
    
    Loads data, trains the FF network using balanced class sampling,
    visualizes metrics, and generates the final submission file.
    """
    base = "/teamspace/studios/this_studio/Barbados_Traffic_Analysis_Challenge_dev"
    train_path = os.path.join(base, "demos/Train.csv")
    test_path = os.path.join(base, "demos/TestInputSegments.csv")
    
    print("Preparing Data...")
    X_train, y_train, X_val, y_val = create_dataset_splits(train_path, val_split=0.2)
    
    # Direct sequential dataset without shuffling or balancing
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(128)

    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weight_dict = dict(zip(classes, weights))

    # Use L2 regularization to prevent weight explosion
    # Define training hyperparameters for the schedule
    global_epochs = 100
    batch_size = 128
    local_layer_epochs = 20 # UPDATED: Tuned value. Changed from 0 to 10 to enable CosineDecay.
    total_batches = len(X_train) // batch_size
    # Ensure total_train_steps is at least 1 to avoid ValueError in CosineDecay
    total_train_steps = max(1, global_epochs * total_batches * local_layer_epochs)
    
    # Global training parameters for layers
    global DROPOUT_RATE, PATIENCE, MIN_DELTA, IS_TRAINING
    DROPOUT_RATE = 0.3
    PATIENCE = 8
    MIN_DELTA = 1e-4
    IS_TRAINING = True

    # Cosine Decay Schedule
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0090513, # UPDATED: Tuned value
        decay_steps=total_train_steps,
        alpha=0.1
    )

    # Use L2 regularization to prevent weight explosion
    reg = keras.regularizers.L2(0.000079024) # UPDATED: Tuned value
    # dims[0] must match the feature length (original features + 4-dim one-hot label)
    input_dim = X_train.shape[1]
    model = FFNetwork(
        dims=[input_dim, 128, 128, 128, 128, 64, 64, 64, 32, 16], # UPDATED: 2 layers of 256 units
        kernel_regularizer=reg,
        learning_rate=lr_schedule,
        use_ema=True,
        ema_overwrite_frequency=100, # UPDATED: Tuned value
        layer_epochs=local_layer_epochs, # UPDATED
        threshold=1.5, # UPDATED: Tuned value
        gamma=1.3, # UPDATED: Tuned value
        dropout_rate=DROPOUT_RATE,
        patience=PATIENCE,
        min_delta=MIN_DELTA
    ) 
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, global_clipnorm=1.0), 
        jit_compile=False
    )
    
    print(f"Training Model ({global_epochs} epochs with Cosine Decay)...")
    history = model.fit(
        train_dataset, 
        validation_data=val_dataset, 
        epochs=global_epochs, 
        class_weight=class_weight_dict,
        verbose=2
    )
    
    # Save the trained model
    models_dir = os.path.join(base, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "ff_model.keras")
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    print("\nVisualizing Metrics...")
    analytics_dir = os.path.join(base, "analytics")
    plot_training_results(history, analytics_dir)
    
    # Switch to inference mode
    IS_TRAINING = False
    
    print("\nInference on TestInputSegments (8-step horizon)...")
    congestion_map = {0: 'free flowing', 1: 'light delay', 2: 'moderate delay', 3: 'heavy delay'}
    test_df = pd.read_csv(test_path)
    
    # Store predictions in a dictionary: (view_label, time_segment_id) -> prediction
    prediction_dict = {}

    for label, group in test_df.groupby('view_label'):
        # Sort by time to ensure sequential continuity
        group = identify_blocks(group)
        
        for b_id, block in group.groupby('block_id'):
            # Predict precisely 8 future steps starting from the end of each block
            feats, _ = get_features_and_labels(block)
            
            # Use the last 15 steps of history as the initial context
            if len(feats) < 15:
                # Fallback: pad with duplicates if block is too short (rare)
                history = [feats[0]] * (15 - len(feats)) + list(feats)
            else:
                history = list(feats[-15:])
            
            # Seg ID is index 3 in the feature vector
            start_id = int(round(history[-1][3] * 5000))
            
            for i in range(1, 9):
                # Current state is the flattened last 15 steps
                current_window = np.array(history[-15:]).flatten()
                
                # Overlay label 0-3 requires 4 zeros buffer
                input_vec = np.zeros(4 + len(current_window), dtype='float32')
                input_vec[4:] = current_window
                
                # Predict enter congestion state
                p_label_idx = model.predict_one(ops.convert_to_tensor(input_vec)).numpy()
                p_label = congestion_map[p_label_idx]
                
                target_id = start_id + i
                prediction_dict[(label, target_id)] = p_label
                
                # Create next step's features to push into history
                next_feat = np.copy(history[-1])
                
                # Update time (assuming 5 minute intervals)
                curr_h = next_feat[0] * 23.0
                curr_m = next_feat[1] * 59.0
                
                curr_m += 5
                if curr_m > 59:
                    curr_m -= 60
                    curr_h = (curr_h + 1) % 24
                
                next_feat[0] = curr_h / 23.0
                next_feat[1] = curr_m / 59.0
                next_feat[3] += (1/5000.0) # Increment segment ID
                # (Note: signaling is assumed constant during forecasting)
                
                history.append(next_feat)

    # Map predictions to SampleSubmission IDs
    print("Mapping 8-step predictions to SampleSubmission template...")
    sample_sub_path = os.path.join(base, "demos/SampleSubmission.csv")
    sample_sub = pd.read_csv(sample_sub_path)
    
    final_targets = []
    for idx, row in sample_sub.iterrows():
        # Parse ID: time_segment_XXX_Label_congestion_enter_rating
        parts = row['ID'].split('_')
        tid = int(parts[2])
        vlabel = parts[3]
        
        # Look up prediction in dictionary, fallback to 'free flowing'
        pred = prediction_dict.get((vlabel, tid), 'free flowing')
        final_targets.append(pred)

    sample_sub['Target'] = final_targets
    sample_sub['Target_Accuracy'] = final_targets
    
    output_path = os.path.join(base, "submissions/submission.csv")
    sample_sub.to_csv(output_path, index=False)
    print(f"\nSubmission file generated: {output_path}")
    print(f"Total rows: {len(sample_sub)}")

if __name__ == "__main__":
    main()