"""
Module implementing the Forward-Forward algorithm for traffic prediction.

This script processes the Barbados Traffic dataset for 'Norman Niles #1' 
and uses the Forward-Forward algorithm to learn representations and 
predict future congestion states.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

# --- Data Preprocessing ---

def load_and_preprocess_data(csv_path: str):
    """Loads and prepares the traffic data for the FF algorithm.
    
    Args:
        csv_path: Path to Train.csv.
        
    Returns:
        X_train, y_train: Input features and target future states.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
        
    df = pd.read_csv(csv_path)
    # Filter for Norman Niles #1 as requested
    df = df[df['view_label'] == 'Norman Niles #1'].copy()
    
    # Sort by time to ensure sequence integrity
    df['video_time'] = pd.to_datetime(df['video_time'])
    df = df.sort_values('video_time')
    
    # Map congestion ratings to numerical values
    congestion_map = {
        'free flowing': 1,
        'light delay': 2,
        'moderate delay': 3,
        'heavy delay': 4
    }
    df['enter_rate'] = df['congestion_enter_rating'].map(congestion_map).fillna(1)
    df['exit_rate'] = df['congestion_exit_rating'].map(congestion_map).fillna(1)
    
    # Feature engineering: Normalize to [0, 1] range
    df['hour'] = df['video_time'].dt.hour / 23.0
    df['minute'] = df['video_time'].dt.minute / 59.0
    df['day'] = df['video_time'].dt.dayofweek / 6.0
    df['enter_norm'] = df['enter_rate'] / 4.0
    df['exit_norm'] = df['exit_rate'] / 4.0
    
    # Input features: Time, Date (day of week), Enter Rate, Exit Rate
    features = df[['hour', 'minute', 'day', 'enter_norm', 'exit_norm']].values
    # Target: Future 5 states (Enter and Exit rates)
    targets = df[['enter_norm', 'exit_norm']].values
    
    # Create windows: Input (t) -> Target (t+1 ... t+5)
    X, y = [], []
    for i in range(len(features) - 5):
        X.append(features[i])
        # Flatten future 5 states (5 intervals * 2 rates = 10 values)
        y.append(targets[i+1 : i+6].flatten())
        
    return np.array(X, dtype='float32'), np.array(y, dtype='float32')


# --- Forward-Forward Model ---

class FFDense(keras.layers.Layer):
    """A single layer trained using the Forward-Forward algorithm."""
    
    def __init__(self, units, threshold=2.0, num_epochs=60, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.threshold = threshold
        self.num_epochs = num_epochs
        # We don't specify input_shape here to allow dynamic building
        self.dense = keras.layers.Dense(units)
        self.relu = keras.layers.ReLU()
        self.optimizer = keras.optimizers.Adam(learning_rate=0.03)

    def call(self, x):
        """Layer activation logic."""
        # Normalize input to prevent magnitude-based learning
        x_norm = tf.norm(x, ord=2, axis=1, keepdims=True) + 1e-4
        x_dir = x / x_norm
        return self.relu(self.dense(x_dir))

    def forward_forward(self, x_pos, x_neg):
        """Local training of the layer weights without backprop through depths."""
        for i in range(self.num_epochs):
            with tf.GradientTape() as tape:
                # Goodness is the mean square of activations
                g_pos = tf.reduce_mean(tf.pow(self.call(x_pos), 2), axis=1)
                g_neg = tf.reduce_mean(tf.pow(self.call(x_neg), 2), axis=1)
                
                # Logistic loss to distinguish positive from negative samples
                loss = tf.reduce_mean(
                    tf.math.log(1 + tf.exp(tf.concat([
                        -g_pos + self.threshold, 
                        g_neg - self.threshold
                    ], 0)))
                )
            
            grads = tape.gradient(loss, self.dense.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.dense.trainable_weights))
            
        # Returning activations for the next layer
        return self.call(x_pos), self.call(x_neg)


class FFNetwork(keras.Model):
    """Network composed of Forward-Forward layers for representation learning."""
    
    def __init__(self, layer_dims, **kwargs):
        super().__init__(**kwargs)
        self.ff_layers = [FFDense(d) for d in layer_dims]
        self.predictor = None

    def train_reps(self, x_pos, x_neg):
        """Sequentially train each FF layer to learn representations."""
        curr_pos, curr_neg = x_pos, x_neg
        for layer in self.ff_layers:
            print(f"  Training FF Layer with {layer.units} units...")
            curr_pos, curr_neg = layer.forward_forward(curr_pos, curr_neg)

    def get_activations(self, x):
        """Computes concatenated activations from all layers as feature vector."""
        activations = []
        curr_x = x
        for layer in self.ff_layers:
            curr_x = layer(curr_x)
            activations.append(curr_x)
        return tf.concat(activations, axis=1)

    def train_predictor_head(self, X, y):
        """Trains a regression head on the fixed FF representations."""
        print("Extracting FF activations for prediction head training...")
        acts = self.get_activations(X)
        
        self.predictor = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(y.shape[1]) # Output 10 nodes forfuture 5 enter/exit states
        ])
        
        self.predictor.compile(optimizer='adam', loss='mse')
        print("Fitting prediction logic...")
        self.predictor.fit(acts, y, epochs=30, batch_size=32, verbose=0)

    def predict_future(self, x):
        """Predicts future congestion states from current input."""
        acts = self.get_activations(x)
        preds = self.predictor(acts).numpy()
        # Scale back and round to mapped values [1, 4]
        return np.clip(np.round(preds * 4.0), 1, 4).astype(int)


# --- Execution ---

def main():
    # User requested to use absolute path
    base_dir = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev"
    csv_path = os.path.join(base_dir, "demos/Train.csv")
    
    print(f"Process started. Environmental check: Tensorflow Conda active.")
    try:
        X, y = load_and_preprocess_data(csv_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Positive data: Real feature sequences
    x_pos = X
    # Negative data: Feature sequences with scrambled correlation (shuffle across features)
    x_neg = np.copy(X)
    for col in range(x_neg.shape[1]):
        np.random.shuffle(x_neg[:, col])
    
    print(f"Dataset Size: {len(X)} samples.")
    print(f"Input Features: {X.shape[1]} (Time, Date, Congestion Enter/Exit)")
    
    # Initialize Model
    model = FFNetwork([512, 512])
    
    print("Step 1: Training Forward-Forward representations (unsupervised)...")
    model.train_reps(x_pos, x_neg)
    
    print("Step 2: Training regression head to predict future 5 states...")
    model.train_predictor_head(X, y)
    
    # Prediction Demo
    test_idx = np.random.randint(0, len(X))
    sample_in = X[test_idx : test_idx + 1]
    actual_out = (y[test_idx] * 4.0).astype(int)
    predicted_out = model.predict_future(sample_in)[0]
    
    # Reshaping for better display (Enter, Exit) pairs
    actual_reshaped = actual_out.reshape(5, 2)
    predicted_reshaped = predicted_out.reshape(5, 2)
    
    print("\n" + "="*50)
    print("TRAFFIC PREDICTION RESULTS (Norman Niles #1)")
    print("="*50)
    print(f"Input Context: {sample_in[0]}")
    print("\nFuture Congestion States (Next 5 Intervals):")
    print("Interval |  Actual (En/Ex)  |  Predicted (En/Ex)")
    print("------------------------------------------")
    for i in range(5):
        print(f"   T + {i+1}  |      {actual_reshaped[i]}      |      {predicted_reshaped[i]}")
    print("="*50)

if __name__ == "__main__":
    main()
