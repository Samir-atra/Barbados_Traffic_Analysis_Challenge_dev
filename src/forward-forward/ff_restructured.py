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

# --- Forward-Forward Classes ---

class FFDense(keras.layers.Layer):
    """A single Forward-Forward Dense layer with local learning.
    
    This layer implements the core FF logic: it trains itself using local 
    goodness scores without global backpropagation. It includes LayerNorm 
    and LeakyReLU for stability and employs Focal Loss to handle hard examples.
    """

    def __init__(self, units, num_epochs=54, kernel_regularizer=None, gamma=1.0337, 
                 threshold=1.143, learning_rate=0.001, use_ema=True, ema_overwrite_frequency=None, **kwargs):
        """Initializes the FFDense layer.

        Args:
            units: Number of hidden units.
            num_epochs: Local training epochs per global epoch.
            kernel_regularizer: Keras regularizer for the dense weights.
            gamma: Focusing parameter for Focal Loss.
            learning_rate: Learning rate for the local optimizer.
            use_ema: Whether to use Exponential Moving Average.
            **kwargs: Standard layer arguments.
        """
        super().__init__(**kwargs)
        self.units = units
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
        self.threshold = threshold
        self.num_epochs = num_epochs
        self.gamma = gamma

    def call(self, x):
        """Forward pass of the layer with unit-norm normalization.

        Args:
            x: Input tensor.

        Returns:
            Normalized and activated layer output.
        """
        x_norm = ops.norm(x, ord=2, axis=1, keepdims=True) + 1e-4
        h = self.dense(x / x_norm)
        return self.activation(self.ln(h))

    def forward_forward(self, x_pos, x_neg, weights=None):
        """Local training logic using the Forward-Forward algorithm.

        Updates layer weights to maximize 'goodness' for positive (real) 
        samples and minimize it for negative (perturbed) samples.

        Args:
            x_pos: Real samples with correct labels overlaid.
            x_neg: Perturbed samples with incorrect labels overlaid.

        Returns:
            A tuple (h_pos, h_neg, loss):
                - h_pos: Layer activations for positive samples.
                - h_neg: Layer activations for negative samples.
                - loss: Mean focal loss for this layer.
        """
        for i in range(self.num_epochs):
            with tf.GradientTape() as tape:
                g_pos = ops.mean(ops.power(self.call(x_pos), 2), 1)
                g_neg = ops.mean(ops.power(self.call(x_neg), 2), 1)
                
                # Standard log-likelihood terms (Binary Cross-Entropy)
                log_pos = ops.log(1 + ops.exp(-g_pos + self.threshold))
                log_neg = ops.log(1 + ops.exp(g_neg - self.threshold))
                
                # Focal Loss weighting: (1 - p)^gamma
                # p_pos = sigmoid(g_pos - threshold) -> 1 - p_pos = sigmoid(threshold - g_pos)
                pt_pos = ops.sigmoid(-g_pos + self.threshold)
                # p_neg = sigmoid(g_neg - threshold) -> targeting 0, so weight is p_neg^gamma
                pt_neg = ops.sigmoid(g_neg - self.threshold)
                
                loss_pos = ops.power(pt_pos, self.gamma) * log_pos
                loss_neg = ops.power(pt_neg, self.gamma) * log_neg
                
                if weights is not None:
                    loss_pos = loss_pos * weights
                    loss_neg = loss_neg * weights
                
                loss = ops.concatenate([loss_pos, loss_neg], 0)
                mean_loss = ops.cast(ops.mean(loss), dtype="float32")
                # Add regularization losses
                if self.dense.losses:
                    mean_loss += ops.sum(self.dense.losses)
                self.loss_metric.update_state([mean_loss])
            grads = tape.gradient(mean_loss, self.dense.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.dense.trainable_weights))
        return ops.stop_gradient(self.call(x_pos)), ops.stop_gradient(self.call(x_neg)), self.loss_metric.result()

class FFNetwork(keras.Model):
    """The full Forward-Forward network model.

    Coordinates layer-wise training, prediction via goodness summation, 
    and custom training/test steps for Keras compatibility.
    """

    def __init__(self, dims, kernel_regularizer=None, learning_rate=0.001, 
                 use_ema=True, ema_overwrite_frequency=None, 
                 layer_epochs=54, threshold=1.143, gamma=1.0337, **kwargs):
        """Initializes the network.

        Args:
            dims: List of dimensions [input_dim, layer1_units, ...].
            kernel_regularizer: Regularizer for all dense layers.
            learning_rate: LR passed to FF layers.
            use_ema: EMA flag passed to FF layers.
            **kwargs: Standard model arguments.
        """
        super().__init__(**kwargs)
        self.loss_var = keras.Variable(0.0, trainable=False)
        self.loss_count = keras.Variable(0.0, trainable=False)
        self.ff_layers = [
            FFDense(d, kernel_regularizer=kernel_regularizer, 
                    learning_rate=learning_rate, use_ema=use_ema,
                    ema_overwrite_frequency=ema_overwrite_frequency,
                    num_epochs=layer_epochs, threshold=threshold, gamma=gamma) 
            for d in dims[1:]
        ]
        self.acc_tracker = keras.metrics.SparseCategoricalAccuracy(name="acc")
        self.f1_tracker = keras.metrics.F1Score(name="f1", average="macro")
        self.precision_tracker = keras.metrics.Precision(name="precision")
        self.recall_tracker = keras.metrics.Recall(name="recall")
        # Explicitly build the model to allow weight saving
        self.build((None, dims[0]))

    @property
    def metrics(self):
        """Returns the model's metrics for tracking."""
        return [self.acc_tracker, self.f1_tracker, self.precision_tracker, self.recall_tracker]

    def call(self, x):
        """Standard forward pass through all layers for Keras tracing."""
        h = x
        for layer in self.ff_layers:
            h = layer(h)
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
        update = xla.dynamic_update_slice(x_zeros, [ops.cast(1.0, x.dtype)], [y_idx])
        return xla.dynamic_update_slice(x, update, [0]), y

    @tf.function
    def predict_batch(self, x):
        """Predicts labels for a batch of inputs using vectorized_map."""
        return ops.vectorized_map(self.predict_one, x)

    def train_step(self, data):
        """Custom training step implementing layer-wise FF updates."""
        if len(data) == 3:
            x, y, weights = data
            weights = ops.cast(weights, "float32")
        else:
            x, y = data
            weights = None

        x_pos, _ = ops.vectorized_map(self.overlay_y_on_x, (x, y))
        
        # Use a deterministic shuffle for the negative pass
        indices = tf.range(start=0, limit=tf.shape(y)[0], dtype=tf.int32)
        shuffled_indices = tf.random.experimental.stateless_shuffle(indices, seed=[SEED, SEED])
        x_neg, _ = ops.vectorized_map(self.overlay_y_on_x, (x, tf.gather(y, shuffled_indices)))
        
        self.loss_var.assign(0.0)
        self.loss_count.assign(0.0)
        
        h_pos, h_neg = x_pos, x_neg
        for layer in self.ff_layers:
            h_pos, h_neg, loss = layer.forward_forward(h_pos, h_neg, weights=weights)
            self.loss_var.assign_add(loss)
            self.loss_count.assign_add(1.0)
        
        # Update training metrics
        y_pred = self.predict_batch(x)
        y_pred_1hot = ops.one_hot(y_pred, 4)
        # Squeeze y to match (batch,) for targets if needed, or update_state handles (batch, 1)
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
        """Custom test step for validation set evaluation."""
        if len(data) == 3:
            x, y, weights = data
            weights = ops.cast(weights, "float32")
        else:
            x, y = data
            weights = None
        
        # Calculate goodness-based loss for validation (layer-wise average)
        x_pos, _ = ops.vectorized_map(self.overlay_y_on_x, (x, y))
        
        # Deterministic shuffle for validation negative pass
        indices = tf.range(start=0, limit=tf.shape(y)[0], dtype=tf.int32)
        shuffled_indices = tf.random.experimental.stateless_shuffle(indices, seed=[SEED, SEED])
        x_neg, _ = ops.vectorized_map(self.overlay_y_on_x, (x, tf.gather(y, shuffled_indices)))
        
        v_loss = 0.0
        h_pos, h_neg = x_pos, x_neg
        for layer in self.ff_layers:
            g_pos = ops.mean(ops.power(layer(h_pos), 2), 1)
            g_neg = ops.mean(ops.power(layer(h_neg), 2), 1)
            
            # Focal Loss terms for validation
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
            # Next layer inputs
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
    base = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev"
    train_path = os.path.join(base, "demos/Train.csv")
    test_path = os.path.join(base, "demos/TestInputSegments.csv")
    
    print("Preparing Data...")
    X_train, y_train, X_val, y_val = create_dataset_splits(train_path)
    
    # Direct sequential dataset without shuffling or balancing
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(64).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64)

    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weight_dict = dict(zip(classes, weights))

    # Use L2 regularization to prevent weight explosion
    # Define training hyperparameters for the schedule
    global_epochs = 30
    batch_size = 64
    local_layer_epochs = 43 # UPDATED: Tuned value
    total_batches = len(X_train) // batch_size
    total_train_steps = global_epochs * total_batches * local_layer_epochs
    
    # Cosine Decay Schedule
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.00074289, # UPDATED: Tuned value
        decay_steps=total_train_steps,
        alpha=0.1
    )

    # Use L2 regularization to prevent weight explosion
    reg = keras.regularizers.L2(1.5576e-05) # UPDATED: Tuned value
    # dims[0] must match the feature length (original features + 4-dim one-hot label)
    input_dim = X_train.shape[1]
    model = FFNetwork(
        dims=[input_dim, 256, 512, 512, 512, 512, 256, 128], # UPDATED: 2 layers of 256 units
        kernel_regularizer=reg,
        learning_rate=lr_schedule,
        use_ema=True,
        ema_overwrite_frequency=1, # UPDATED: Tuned value
        layer_epochs=local_layer_epochs, # UPDATED
        threshold=1.0395, # UPDATED: Tuned value
        gamma=2.4788 # UPDATED: Tuned value
    ) 
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, global_clipnorm=1.0), 
        jit_compile=False,
        metrics=["acc", "f1", "precision", "recall"]
    )
    
    print(f"Training Model ({global_epochs} epochs with Cosine Decay)...")
    history = model.fit(
        train_dataset, 
        validation_data=val_dataset, 
        epochs=global_epochs, 
        class_weight=class_weight_dict,
        verbose=2
    )
    
    print("\nVisualizing Metrics...")
    analytics_dir = os.path.join(base, "analytics")
    plot_training_results(history, analytics_dir)
    
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
