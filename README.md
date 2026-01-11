# Barbados Traffic Analysis Challenge 🚗🚦

This repository contains the implementation of a **Forward-Forward (FF)** neural network architecture designed for sequential traffic congestion prediction in the [Barbados Traffic Analysis Challenge](https://zindi.africa/competitions/barbados-traffic-analysis-challenge).

The objective is to predict the congestion level (4 classes) for the next 8 time segments based on historical traffic patterns and signaling states.

## 🚀 Key Features

- **Forward-Forward Algorithm**: Implements Hinton's FF architecture, replacing traditional Backpropagation with local "goodness" maximization.
- **Temporal Context**: Leverages a **15-step sliding window** (195 features) to capture short-term traffic trends and periodicities.
- **Robust Training**:
  - **Focal Loss**: Specifically tuned to handle the severe class imbalance in traffic data.
  - **Cosine Decay**: Implements a smooth learning rate schedule calibrated for local layer updates.
  - **Exponential Moving Average (EMA)**: Ensures weight stability across training iterations.
- **Kaggle Optimized**: Includes a dedicated Hyperband tuning kernel optimized for Kaggle GPU environments.

## 📁 Repository Structure

```tree
.
├── analytics/              # Visualization of training metrics (Loss, F1, Acc)
├── demos/                  # Dataset files (Train.csv, TestInputSegments.csv)
├── docs/                   # Experimentation notes and hyperparameter logs
├── kaggle_kernel/          # Automated tuning script and metadata for Kaggle push
│   └── ff_hyperband_kernel.py
├── submissions/            # Generated submission.csv files
└── src/
    └── forward-forward/
        └── ff_restructured.py  # Primary training and autoregressive inference script
```

## 🧠 Methodology

### Forward-Forward (FF) Architecture
Unlike standard neural networks that use a backward pass to distribute errors, this model uses a **Local Goodness** objective. Each layer independently learns to:
1. **Maximize goodness** for "Positive" data (Real features + Correct label overlay).
2. **Minimize goodness** for "Negative" data (Real features + Incorrect label overlay).

### Data Pipeline
- **Input Dimensions**: 199 (15 time steps × 13 features + 4-dim label buffer).
- **Label Buffer**: The label is one-hot encoded and overlaid on the first 4 dimensions of the input vector, allowing the model to distinguish between different "hypothesized" states.
- **Inference**: Uses an **Autoregressive Forecaster** where the predicted state $t+1$ is recycled back into the 15-step history window to predict $t+2$ through $t+8$.

## 🛠️ Usage

### Prerequisites
- Python 3.10+
- TensorFlow / Keras 3.x
- Pandas, NumPy, Matplotlib

### Local Training
To train the model on your local machine using the current best hyperparameters:
```bash
python src/forward-forward/ff_restructured.py
```

### Hyperparameter Tuning (Kaggle)
The tuning script is designed to run in a Kaggle environment via the `kaggle` CLI:
```bash
cd kaggle_kernel
kaggle kernels push
```

## 📊 Current Metrics
Based on the latest runs with a sequence length of 15:
- **Macro F1 Score**: ~0.37 (Targeting minority classes: Light, Moderate, Heavy Delay).
- **Inference Horizon**: 8 sequential segments (Approx. 40 minutes of traffic).

## 📝 License
This project is licensed under the Apache 2.0 License.
