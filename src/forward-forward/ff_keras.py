import os
import matplotlib
matplotlib.use('Agg') # Set backend to Agg for headless environments

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
from keras import ops
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import random
from tensorflow.compiler.tf2xla.python import xla

# --- Data Loading and Preprocessing ---

def load_traffic_data(csv_path):
    """Loads and preprocesses traffic data for the FF algorithm."""
    df = pd.read_csv(csv_path)
    
    # Filter columns
    df = df[['video_time', 'date', 'view_label', 'congestion_enter_rating', 'congestion_exit_rating']].copy()
    
    # Date/Time features
    df['video_time'] = pd.to_datetime(df['video_time'])
    df['hour'] = df['video_time'].dt.hour / 23.0
    df['minute'] = df['video_time'].dt.minute / 59.0
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek / 6.0
    
    # View label encoding (4 locations)
    view_map = {label: i for i, label in enumerate(df['view_label'].unique())}
    df['view_id'] = df['view_label'].map(view_map)
    # One-hot encoding for view_label features
    view_one_hot = pd.get_dummies(df['view_id'], prefix='view').astype(float)
    
    # Congestion Label Mapping
    congestion_map = {
        'free flowing': 0,
        'light delay': 1,
        'moderate delay': 2,
        'heavy delay': 3
    }
    df['enter_id'] = df['congestion_enter_rating'].map(congestion_map).fillna(0).astype(int)
    df['exit_id'] = df['congestion_exit_rating'].map(congestion_map).fillna(0).astype(int)
    
    # Joint label for classification (4x4 = 16 possibilities)
    df['joint_label'] = df['enter_id'] * 4 + df['exit_id']
    
    # Combine features
    # Features: [hour, minute, day_of_week, view_0, view_1, view_2, view_3]
    X_feats = df[['hour', 'minute', 'day_of_week']].values
    X_view = view_one_hot.values
    X = np.concatenate([X_feats, X_view], axis=1).astype('float32')
    
    # Prepend 16 zeros for the label overlay (total 16 + 7 = 23)
    X_padded = np.zeros((X.shape[0], 16 + X.shape[1]), dtype='float32')
    X_padded[:, 16:] = X
    
    y = df['joint_label'].values.astype('int')
    
    return train_test_split(X_padded, y, test_size=0.2, random_state=42)

# --- FF Model Components ---

class FFDense(keras.layers.Layer):
    def __init__(
        self,
        units,
        init_optimizer,
        loss_metric,
        num_epochs=100,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(
            units=units,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
        self.relu = keras.layers.ReLU()
        self.optimizer = init_optimizer()
        self.loss_metric = loss_metric
        self.threshold = 1.5
        self.num_epochs = num_epochs

    def call(self, x):
        x_norm = ops.norm(x, ord=2, axis=1, keepdims=True)
        x_norm = x_norm + 1e-4
        x_dir = x / x_norm
        res = self.dense(x_dir)
        return self.relu(res)

    def forward_forward(self, x_pos, x_neg):
        for i in range(self.num_epochs):
            with tf.GradientTape() as tape:
                g_pos = ops.mean(ops.power(self.call(x_pos), 2), 1)
                g_neg = ops.mean(ops.power(self.call(x_neg), 2), 1)

                loss = ops.log(
                    1
                    + ops.exp(
                        ops.concatenate(
                            [-g_pos + self.threshold, g_neg - self.threshold], 0
                        )
                    )
                )
                mean_loss = ops.cast(ops.mean(loss), dtype="float32")
                self.loss_metric.update_state([mean_loss])
            gradients = tape.gradient(mean_loss, self.dense.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.dense.trainable_weights))
        return (
            ops.stop_gradient(self.call(x_pos)),
            ops.stop_gradient(self.call(x_neg)),
            self.loss_metric.result(),
        )

class FFNetwork(keras.Model):
    def __init__(
        self,
        dims,
        init_layer_optimizer=lambda: keras.optimizers.Adam(learning_rate=0.03),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.init_layer_optimizer = init_layer_optimizer
        self.loss_var = keras.Variable(0.0, trainable=False, dtype="float32")
        self.loss_count = keras.Variable(0.0, trainable=False, dtype="float32")
        self.layer_list = [keras.Input(shape=(dims[0],))]
        self.metrics_built = False
        for d in range(len(dims) - 1):
            self.layer_list += [
                FFDense(
                    dims[d + 1],
                    init_optimizer=self.init_layer_optimizer,
                    loss_metric=keras.metrics.Mean(),
                )
            ]
        # Trackers for the requested metrics
        self.acc_tracker = keras.metrics.SparseCategoricalAccuracy(name="acc")
        self.f1_tracker = keras.metrics.F1Score(name="f1", average="macro")

    @property
    def metrics(self):
        # Return trackers so Keras handles them in the progress bar
        return [self.acc_tracker, self.f1_tracker]

    @tf.function(reduce_retracing=True)
    def overlay_y_on_x(self, data):
        X_sample, y_sample = data
        X_zeros = ops.zeros([16], dtype=X_sample.dtype)
        val = ops.cast(1.0, dtype=X_sample.dtype)
        X_update = xla.dynamic_update_slice(X_zeros, [val], [y_sample])
        X_sample = xla.dynamic_update_slice(X_sample, X_update, [0])
        return X_sample, y_sample

    @tf.function(reduce_retracing=True)
    def predict_one_sample(self, x):
        goodness_per_label = []
        for label in range(16):
            h, _ = self.overlay_y_on_x(data=(x, label))
            h = ops.reshape(h, [1, -1])
            layer_goodness = []
            for layer in self.layers:
                if isinstance(layer, FFDense):
                    h = layer(h)
                    layer_goodness += [ops.mean(ops.power(h, 2), 1)]
            goodness_per_label += [ops.expand_dims(ops.sum(layer_goodness), 0)]
        goodness_per_label = ops.concatenate(goodness_per_label, 0)
        return ops.cast(ops.argmax(goodness_per_label), dtype="int32")

    def predict(self, data):
        preds = ops.vectorized_map(self.predict_one_sample, data)
        return np.asarray(preds, dtype=int)

    @tf.function(jit_compile=True)
    def train_step(self, data):
        x, y = data
        
        # 1. Generate positive and negative samples
        x_pos, _ = ops.vectorized_map(self.overlay_y_on_x, (x, y))
        random_y = tf.random.shuffle(y)
        x_neg, _ = ops.vectorized_map(self.overlay_y_on_x, (x, random_y))

        h_pos, h_neg = x_pos, x_neg
        self.loss_var.assign(0.0)
        self.loss_count.assign(0.0)

        # 2. Local layer-wise training
        for layer in self.layers:
            if isinstance(layer, FFDense):
                h_pos, h_neg, loss = layer.forward_forward(h_pos, h_neg)
                self.loss_var.assign_add(loss)
                self.loss_count.assign_add(1.0)
        
        # 3. Calculate internal predictions for metrics update
        y_pred = ops.vectorized_map(self.predict_one_sample, x)
        
        # SparseCategoricalAccuracy expects labels as integers and predictions as probabilities/one-hot
        y_pred_one_hot = ops.one_hot(y_pred, 16)
        self.acc_tracker.update_state(y, y_pred_one_hot)
        
        # F1 Score requires one-hot encoded targets and predictions
        y_one_hot = ops.one_hot(y, 16)
        self.f1_tracker.update_state(y_one_hot, y_pred_one_hot)

        mean_res = ops.divide(self.loss_var, self.loss_count)
        return {
            "loss": mean_res, 
            "acc": self.acc_tracker.result(), 
            "f1": self.f1_tracker.result()
        }

# --- Execution ---

csv_path = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev/demos/Train.csv"
x_train, x_test, y_train, y_test = load_traffic_data(csv_path)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(x_test))

model = FFNetwork(dims=[23, 512, 512])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.03),
    jit_compile=False,
    metrics=["acc", "f1"],
)

epochs = 100
print("Training FF Model on Traffic Data...")
history = model.fit(train_dataset, epochs=epochs)

print("Evaluating on Test Set...")
preds = model.predict(ops.convert_to_tensor(x_test))
preds = preds.flatten()

enter_actual = y_test // 4
exit_actual = y_test % 4
enter_pred = preds // 4
exit_pred = preds % 4

joint_acc = accuracy_score(y_test, preds)
enter_acc = accuracy_score(enter_actual, enter_pred)
exit_acc = accuracy_score(exit_actual, exit_pred)

print(f"\n--- Results ---")
print(f"Joint Accuracy: {joint_acc*100:.2f}%")
print(f"Enter Accuracy: {enter_acc*100:.2f}%")
print(f"Exit Accuracy:  {exit_acc*100:.2f}%")

plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"])
plt.title("Forward-Forward Loss over Training (Traffic)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev/analytics/ff_keras_traffic_loss.png")
print("Loss plot saved to analytics/ff_keras_traffic_loss.png")
