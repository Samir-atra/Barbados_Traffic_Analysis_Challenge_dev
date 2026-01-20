#!/usr/bin/env python
# coding: utf-8

# # Traffic Congestion Prediction: MobileNet + Echo State Network (ESN)
# 
# This notebook implements a **training-free** pipeline for video classification, suitable for cloud environments (Kaggle/Colab).
# 
# ### Approach (Option 4):
# 1.  **Spatial Features**: Use a **Frozen Pre-trained MobileNetV2** to extract high-level visual features (1280-dim) from each frame. No backpropagation is performed on the CNN.
# 2.  **Temporal Features**: Use an **Echo State Network (ESN)** (Reservoir Computing) to model the temporal evolution of these features over time.
# 3.  **Classification**: Train a linear readout (Ridge Regression) to predict traffic congestion levels.
# 
# **Advantages**: 
# - **Extremely Fast Training**: No gradient descent, just one-shot matrix solution.
# - **Low Compute**: Can perform reasonably well without high-end GPUs for training.
# - **Video-Native**: Processes raw video frames.

# In[ ]:


# Install dependencies
get_ipython().system('pip install decord scikit-learn polars')


# In[ ]:


import os
import cv2
import numpy as np
import pandas as pd
import polars as pl
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import decord
from decord import VideoReader, cpu, gpu

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print(f"TensorFlow GPUs Available: {len(gpus)}")
USE_GPU_DECORD = len(gpus) > 0


# ## 1. Configuration & Paths
# Please set the following paths to your dataset locations.

# In[ ]:


# --- USER CONFIGURATION ---
BASE_DIR = '/kaggle/input/barbados-traffic-analysis-challenge' # Example
VIDEO_DIR = '/kaggle/input/barbados-traffic-analysis-challenge/videos' # Example
TRAIN_CSV = os.path.join(BASE_DIR, 'Train.csv')
TEST_CSV = os.path.join(BASE_DIR, 'TestInputSegments.csv')
SAMPLE_SUB = os.path.join(BASE_DIR, 'SampleSubmission.csv')

# Model Config
IMG_SIZE = (224, 224)
SEQ_LENGTH = 30         # Number of frames to extract per video (downsampled)
RESERVOIR_DIM = 1000    # ESN Reservoir Size
SPECTRAL_RADIUS = 0.9
LEAK_RATE = 0.2
RIDGE_ALPHA = 1.0
# --------------------------


# ## 2. Feature Extraction (MobileNetV2)
# We load a MobileNetV2 pre-trained on ImageNet, remove the top classification layer, and use global average pooling to get a 1280-dimensional vector for each frame.

# In[ ]:


def build_feature_extractor():
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        pooling='avg',
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False  # Freeze weights
    return base_model

feat_extractor = build_feature_extractor()
print("Feature Extractor Loaded: MobileNetV2 (Frozen)")


# ## 3. Video Processing Utils
# Functions to read video frames and extract features.

# In[ ]:


def extract_frames(video_path, seq_len=SEQ_LENGTH):
    if not os.path.exists(video_path):
        return np.zeros((seq_len, *IMG_SIZE, 3)) # Return silent zero-padding if missing
        
    try:
        ctx = gpu(0) if USE_GPU_DECORD else cpu(0)
        vr = VideoReader(video_path, ctx=ctx)
        total_frames = len(vr)
        
        # Uniform sampling
        indices = np.linspace(0, total_frames - 1, seq_len).astype(int)
        frames = vr.get_batch(indices).asnumpy()
        
        # Resize and Preprocess
        processed_frames = []
        for frame in frames:
            frame = cv2.resize(frame, IMG_SIZE)
            processed_frames.append(frame)
            
        processed_frames = np.array(processed_frames)
        processed_frames = preprocess_input(processed_frames) # MobileNet preprocessing (-1 to 1)
        return processed_frames
        
    except Exception as e:
        print(f"Error reading {video_path}: {e}")
        return np.zeros((seq_len, *IMG_SIZE, 3))

def get_video_features(video_path):
    frames = extract_frames(video_path)
    # Batch prediction (T, 224, 224, 3) -> (T, 1280)
    features = feat_extractor.predict(frames, verbose=0)
    return features


# ## 4. Echo State Network Class
# A pure NumPy implementation of ESN for sequence classification.

# In[ ]:


class ELM_ESN_Classifier:
    def __init__(self, input_dim=1280, res_dim=1000, rho=0.9, leak=0.2, alpha=1.0):
        self.res_dim = res_dim
        self.rho = rho
        self.leak = leak
        self.alpha = alpha
        
        # Initialize weights (Training Free)
        rng = np.random.RandomState(42)
        self.W_in = rng.uniform(-1, 1, (res_dim, input_dim))
        # Sparse recurrent weights
        self.W_res = rng.uniform(-1, 1, (res_dim, res_dim))
        mask = rng.rand(res_dim, res_dim) > 0.95 # 5% sparsity
        self.W_res[mask] = 0
        
        # Spectral Normalization
        eigenvalues = np.linalg.eigvals(self.W_res)
        max_eig = np.max(np.abs(eigenvalues))
        self.W_res *= (self.rho / max_eig)
        
        self.readout = Ridge(alpha=alpha)
        
    def get_states(self, inputs):
        # inputs: (T, input_dim)
        T = inputs.shape[0]
        states = np.zeros((T, self.res_dim))
        x = np.zeros(self.res_dim)
        
        for t in range(T):
            u = inputs[t]
            # ESN State Update
            pre_act = np.dot(self.W_in, u) + np.dot(self.W_res, x)
            x_tilde = np.tanh(pre_act)
            x = (1 - self.leak) * x + self.leak * x_tilde
            states[t] = x
            
        # Return final state (or average state) as sequence representation
        # Here we use the mean state of the sequence
        return np.mean(states, axis=0)
    
    def fit(self, X_features, y):
        # X_features: List of (T, 1280) arrays
        # y: Array of labels
        X_states = []
        for feats in X_features:
            state = self.get_states(feats)
            X_states.append(state)
        
        X_states = np.array(X_states)
        self.readout.fit(X_states, y)
        
    def predict(self, X_features):
        X_states = []
        for feats in X_features:
            state = self.get_states(feats)
            X_states.append(state)
        return self.readout.predict(np.array(X_states))


# ## 5. Load Data & Train
# We will load video paths, labels, and run the pipeline.

# In[ ]:


# Load Train CSV
df = pd.read_csv(TRAIN_CSV)

# Fix Video Paths function
def extract_camera_path(video_col):
    # Assumes format in original notebook
    # Customize this based on actual file structure
    # E.g. "normanniles1/normanniles1_2025...mp4" -> "videos/normanniles1_...mp4"
    # For now, we assume simple join with VIDEO_DIR
    filename = os.path.basename(video_col)
    return os.path.join(VIDEO_DIR, filename)

df['video_full_path'] = df['videos'].apply(extract_camera_path)

# Map Labels
congestion_map = {'free flowing': 0, 'light delay': 1, 'moderate delay': 2, 'heavy delay': 3}
df['label_code'] = df['congestion_enter_rating'].map(congestion_map).fillna(0).astype(int)

# Train/Val Split (Video Level)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_code'])

print(f"Train Samples: {len(train_df)}, Val Samples: {len(val_df)}")


# In[ ]:


def extract_dataset_features(dataframe):
    features_list = []
    labels_list = []
    
    total = len(dataframe)
    for idx, row in dataframe.reset_index().iterrows():
        path = row['video_full_path']
        label = row['label_code']
        
        if idx % 50 == 0: print(f"Processing {idx}/{total}...")
        
        # MobileNet Feature Extraction
        feats = get_video_features(path)
        
        features_list.append(feats)
        labels_list.append(label)
        
    return features_list, np.array(labels_list)

print("Extracting TRAIN Features (this may take time)...")
X_train_feats, y_train = extract_dataset_features(train_df)

print("Extracting VAL Features...")
X_val_feats, y_val = extract_dataset_features(val_df)


# In[ ]:


# Initialize and Train ESN
esn = ELM_ESN_Classifier(
    res_dim=RESERVOIR_DIM, 
    rho=SPECTRAL_RADIUS, 
    leak=LEAK_RATE, 
    alpha=RIDGE_ALPHA
)

print("Training ESN Reservoir...")
esn.fit(X_train_feats, y_train)
print("Training Complete.")


# In[ ]:


# Evaluate
print("Evaluating...")
val_preds = esn.predict(X_val_feats)
# Round predictions to nearest class (Ridge Regression gives floats)
val_preds_class = np.round(np.clip(val_preds, 0, 3)).astype(int)

acc = accuracy_score(y_val, val_preds_class)
f1 = f1_score(y_val, val_preds_class, average='macro')

print(f"Validation Accuracy: {acc:.4f}")
print(f"Validation F1 Score: {f1:.4f}")


# ## 6. Inference on Test Set
# This section generates the submission file.

# In[ ]:


test_df = pd.read_csv(TEST_CSV)

# Adjust this mapping to match test CSV structure
# Assuming test_df has a 'video' columns or similar ID
# Note: The test set often comes as segments. You will need to map segments to available video files.
# For this notebook, we assume we process available videos and map predictions.

sample_sub = pd.read_csv(SAMPLE_SUB)
print("Sample Submission:", sample_sub.head())

# Placeholder for actual test video processing logic
# Here you would iterate through test sample IDs, find the corresponding video, extract features, and predict.
print("Inference placeholder - Implement specific test mapping logic here based on competition rules.")

