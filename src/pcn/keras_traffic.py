"""Keras Neural Network for Traffic Prediction.

This script implements a deep neural network using Keras/TensorFlow for
traffic congestion classification with the goal of achieving 90%+ accuracy.

Features:
- Deep residual architecture
- Learning rate scheduling
- Data augmentation
- Proper batch normalization
- Early stopping with patience
"""

import os
import sys
import numpy as np
import pandas as pd

# Add data_processing to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
from chunked_data_loader import (
    get_features_and_labels,
    identify_blocks,
)

# Set seeds
SEED = 42
np.random.seed(SEED)

# TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.random.set_seed(SEED)
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def create_raw_sequences(csv_path, val_split=0.2, seq_len=15):
    """Processes CSV into 3D sequential samples."""
    df = pd.read_csv(csv_path)
    train_X, train_y = [], []
    val_X, val_y = [], []
    
    print(f"Loading {len(df)} rows from {csv_path}...")
    for label, group in df.groupby('view_label'):
        group = identify_blocks(group)
        
        for b_id, block in group.groupby('block_id'):
            if len(block) < seq_len + 1:
                continue
            
            n_block = len(block)
            n_train_rows = int(n_block * (1 - val_split))
            
            train_block = block.iloc[:n_train_rows]
            val_block = block.iloc[n_train_rows:]
            
            def make_seqs(sub_block):
                feats, labels = get_features_and_labels(sub_block)
                X_w, y_w = [], []
                for i in range(len(feats) - seq_len):
                    X_w.append(feats[i : i + seq_len])
                    y_w.append(labels[i + seq_len])
                return X_w, y_w
            
            if len(train_block) >= seq_len + 1:
                tx, ty = make_seqs(train_block)
                train_X.extend(tx)
                train_y.extend(ty)
            
            if len(val_block) >= seq_len + 1:
                vx, vy = make_seqs(val_block)
                val_X.extend(vx)
                val_y.extend(vy)

    return np.array(train_X), np.array(train_y), np.array(val_X), np.array(val_y)


def build_model(input_shape, num_classes=4):
    """Build a deep residual neural network."""
    inputs = keras.Input(shape=input_shape)
    
    # Flatten the sequence
    x = layers.Flatten()(inputs)
    
    # Dense block 1
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)
    
    # Dense block 2 with residual
    shortcut = layers.Dense(256)(x)
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    # Dense block 3 with residual
    shortcut = layers.Dense(128)(x)
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    # Dense block 4
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model


def build_lstm_model(input_shape, num_classes=4):
    """Build an LSTM model for sequential data."""
    inputs = keras.Input(shape=input_shape)
    
    # Bidirectional LSTM layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, 
                                          kernel_regularizer=regularizers.l2(1e-4)))(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False,
                                          kernel_regularizer=regularizers.l2(1e-4)))(x)
    x = layers.Dropout(0.3)(x)
    
    # Dense layers
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model


def main():
    """Main training function."""
    base = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev"
    train_path = os.path.join(base, "demos/Train.csv")
    test_path = os.path.join(base, "demos/TestInputSegments.csv")
    sample_sub_path = os.path.join(base, "demos/SampleSubmission.csv")
    
    print("Loading data...")
    X_train, y_train, X_val, y_val = create_raw_sequences(train_path, val_split=0.15)
    
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    print(f"Train classes: {np.bincount(y_train)}")
    print(f"Val classes: {np.bincount(y_val)}")
    
    # Normalize data
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    
    # Class weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    print(f"Class weights: {class_weight_dict}")
    
    # Build model - try LSTM for sequential data
    print("\nBuilding LSTM model...")
    model = build_lstm_model(X_train.shape[1:], num_classes=4)
    
    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
    ]
    
    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n--- Final Evaluation ---")
    train_pred = np.argmax(model.predict(X_train, verbose=0), axis=1)
    val_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    
    print("Training Data:")
    print(f"  Accuracy: {accuracy_score(y_train, train_pred):.4f}")
    print(f"  F1 Macro: {f1_score(y_train, train_pred, average='macro'):.4f}")
    
    print("Validation Data:")
    print(f"  Accuracy:  {accuracy_score(y_val, val_pred):.4f}")
    print(f"  F1 Macro:  {f1_score(y_val, val_pred, average='macro'):.4f}")
    print(f"  Precision: {precision_score(y_val, val_pred, average='macro', zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_val, val_pred, average='macro', zero_division=0):.4f}")
    
    # Inference
    print("\nStarting Inference...")
    congestion_map = {0: 'free flowing', 1: 'light delay', 
                      2: 'moderate delay', 3: 'heavy delay'}
    test_df = pd.read_csv(test_path)
    prediction_dict = {}
    
    for label, group in test_df.groupby('view_label'):
        group = identify_blocks(group)
        for b_id, block in group.groupby('block_id'):
            feats, _ = get_features_and_labels(block)
            
            if len(feats) < 15:
                history_feats = [feats[0]] * (15 - len(feats)) + list(feats)
            else:
                history_feats = list(feats[-15:])
            
            start_id = int(round(history_feats[-1][10] * 5000))
            
            for i in range(1, 9):
                current_seq = np.array(history_feats[-15:]).reshape(1, 15, -1)
                current_seq = (current_seq - mean) / std
                
                p_idx = np.argmax(model.predict(current_seq, verbose=0), axis=1)[0]
                p_label = congestion_map[p_idx]
                prediction_dict[(label, start_id + i)] = p_label
                
                next_feat = np.copy(history_feats[-1])
                curr_h = next_feat[8] * 23.0
                curr_m = next_feat[9] * 59.0
                curr_m += 5
                if curr_m > 59:
                    curr_m -= 60
                    curr_h = (curr_h + 1) % 24
                next_feat[8] = curr_h / 23.0
                next_feat[9] = curr_m / 59.0
                next_feat[10] += 1/5000.0
                history_feats.append(next_feat)

    print("Mapping predictions...")
    sample_sub = pd.read_csv(sample_sub_path)
    final_targets = []
    
    for idx, row in sample_sub.iterrows():
        parts = row['ID'].split('_')
        tid = int(parts[2])
        vlabel = parts[3]
        final_targets.append(prediction_dict.get((vlabel, tid), 'free flowing'))
        
    sample_sub['Target'] = final_targets
    sample_sub['Target_Accuracy'] = final_targets
    
    output_dir = os.path.join(base, "submissions")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "submission_keras_lstm.csv")
    sample_sub.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
