"""Module for inference using the trained Forward-Forward algorithm model.

This script loads a pre-trained Forward-Forward (FF) neural network model
and uses it to generate traffic congestion predictions for the test dataset.
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
    """Extracts and encodes features and labels from a traffic dataframe (Polars + JAX)."""
    # Ensure date/time columns are proper types
    df = df.with_columns([
        pl.col("video_time").str.to_datetime(),
        pl.col("date").str.to_datetime()
    ])

    df = df.with_columns([
        (pl.col("video_time").dt.hour() / 23.0).alias("hour"),
        (pl.col("video_time").dt.minute() / 59.0).alias("minute"),
        ((pl.col("date").dt.weekday() - 1) / 6.0).alias("day_of_week") # Mon=1 -> 0
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
    
    return features, labels

def identify_blocks(group):
    """Sorts by time_segment_id and identifies continuous sequential blocks (Polars)."""
    group = group.sort("time_segment_id")
    diff = group["time_segment_id"].diff().fill_null(1)
    is_break = (diff != 1).cast(pl.Int32)
    group = group.with_columns(is_break.cum_sum().alias("block_id"))
    return group

# --- Forward-Forward Classes (Must match Training Definitions) ---

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
        # Training logic not strictly needed for inference, but kept for class completeness
        return ops.stop_gradient(self.call(x_pos)), ops.stop_gradient(self.call(x_neg)), 0.0

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
        pass # Not used in inference

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
        pass # Not used in inference

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
        # Metrics not strictly needed for inference only, but config requires them if they were saved
        self.acc_tracker = keras.metrics.SparseCategoricalAccuracy(name="acc")
        self.f1_tracker = keras.metrics.F1Score(name="f1", average="macro")
        self.precision_tracker = MacroPrecision(name="precision")
        self.recall_tracker = MacroRecall(name="recall")
        
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

    def call(self, x):
        h = x
        for layer in self.ff_layers:
            h = layer(h)
        return h

    def overlay_y_on_x(self, data):
        x, y = data
        x_zeros = ops.zeros([4], dtype=x.dtype)
        y_idx = ops.reshape(ops.cast(y, "int32"), [])
        update = xla.dynamic_update_slice(x_zeros, [ops.cast(10.0, x.dtype)], [y_idx])
        return xla.dynamic_update_slice(x, update, [0]), y

    @tf.function
    def predict_batch(self, x):
        return ops.vectorized_map(self.predict_one, x)

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

# --- Main Inference ---

def main():
    """Main execution block for inference."""
    # Kaggle specific paths
    test_path = "/kaggle/input/barbados-traffic-analysis-challenge/TestInputSegments.csv"
    sample_sub_path = "/kaggle/input/barbados-traffic-analysis-challenge/SampleSubmission.csv"
    
    # Path where model is expected
    base_out = "/kaggle/working"
    model_path = os.path.join(base_out, "ff_model.keras") 
    
    submissions_dir = os.path.join(base_out, "submissions")
    os.makedirs(submissions_dir, exist_ok=True)
    
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}. Please train the model first.")
        return

    # Load the model directly. Keras uses the registered classes to reconstruct it.
    model = keras.models.load_model(model_path)
    print("Model loaded successfully.")
    
    print("\nInference on TestInputSegments (8-step horizon)...")
    congestion_map = {0: 'free flowing', 1: 'light delay', 2: 'moderate delay', 3: 'heavy delay'}
    test_df = pl.read_csv(test_path)
    
    # Store predictions in a dictionary: (view_label, time_segment_id) -> prediction
    prediction_dict = {}

    partitions = test_df.partition_by("view_label")

    for group in partitions:
        group = identify_blocks(group)
        label = group["view_label"][0]
        
        block_partitions = group.partition_by("block_id")

        for block in block_partitions:
            feats, _ = get_features_and_labels(block)
            
            if len(feats) < 15:
                # Handle short blocks
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
                
                # Predict
                p_label_idx = model.predict_one(ops.convert_to_tensor(input_vec)).numpy()
                p_label = congestion_map[p_label_idx]
                
                target_id = start_id + i
                prediction_dict[(label, target_id)] = p_label
                
                # Update history
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

if __name__ == "__main__":
    main()
