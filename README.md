# Barbados Traffic Analysis Challenge 🚗🚦

This repository for the research paper: [Forward-Pass Only Deep Echo State Networks for Trac Congestion Prediction: A Case Study on the Barbados Trac Analysis Challenge](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6730183)
And soon a book will be published with more!

This repository contains the solution for the [Barbados Traffic Analysis Challenge](https://zindi.africa/competitions/barbados-traffic-analysis-challenge). The goal is to predict traffic congestion levels (4 classes) for the next 8 time segments based on historical traffic patterns and signaling states.

**Key Constraint**: The competition requires that the training and inference models **must not use backpropagation**. This repository explores alternative learning paradigms like **Echo State Networks (ESN)**, **Extreme Learning Machines (ELM)**, and **Forward-Forward (FF)** algorithms.

## 🚀 Key Features

*   **Deep Echo State Network (DeepESN)**: A stacked reservoir computing architecture that captures temporal dynamics without backpropagation.
    *   Optimized hyperparameters: Spectral Radius, Leak Rate, Reservoir Dimension.
    *   Forward-pass only training using Ridge Regression.
*   **Extreme Learning Machine (ELM)**: Single-hidden layer feedforward networks with randomized weights and analytical output weight calculation.
*   **Forward-Forward Algorithm**: An implementation of Hinton's local goodness maximization (archived).
*   **Video Analysis**: Preliminary exploration of traffic density estimation using computer vision (MediaPipe, MobileNet) to correlate video feeds with congestion labels.
*   **Robust Data Pipeline**:
    *   Handling sequence length variations with chunking and block processing.
    *   Focal Loss Integration (for applicable components).
    *   Signaling feature engineering (None, Low, Medium, High).

## 📁 Repository Structure

```tree
.
├── analytics/              # Visualization of training metrics
├── datasets/               # Raw and processed datasets
├── docs/                   # Experimentation notes and logs
├── src/
│   ├── esn_elm/            # Active: DeepESN and ELM implementations
│   │   ├── max_deep_esn.py # Main DeepESN training and inference script
│   │   ├── esn_traffic.py  # ESN baseline
│   │   └── run_deep_esn_tuning.py # Hyperparameter tuning for DeepESN
│   ├── forward-forward/    # Archived: FF algorithm implementation
│   ├── video_pipeline/     # Video processing and object counting experiments
│   └── data_processing/    # Data loading and utility scripts
├── submissions/            # Generated submission files
└── tests/                  # Unit tests
```

## 🧠 Methodology

### Deep Echo State Network (DeepESN)
The core model is a DeepESN, which consists of a stack of non-linear reservoir layers.
1.  **Input**: Sequential traffic data (features + mapped targets).
2.  **Reservoir**: Multiple layers of recurrently connected neurons with fixed, randomized weights. Each layer feeds into the next, creating a deep temporal representation.
3.  **Readout**: A linear readout layer trained via **Ridge Regression** (closed-form solution) to map the reservoir states to the target traffic classes.
4.  **Inference**: Autoregressive forecasting where predictions are fed back to generate the full 8-step horizon.

### Video Analysis (Exploratory)
We explored using object detection (efficientnet, various CNNs) to count vehicles in video feeds to ground-truth the "Traffic Density" labels. While resource-intensive, this provided insights into the correlation between visual traffic flow and the provided labels.

## 🛠️ Usage

### Prerequisites
*   Python 3.10+
*   TensorFlow / Keras 3.x (for data handling/utilities)
*   Polars (for fast tabular data loading)
*   Scikit-learn, Numpy, Matplotlib

### Running Default Training (DeepESN)
To train the DeepESN model using the current best hyperparameters:

```bash
python src/esn_elm/max_deep_esn.py
```

### Hyperparameter Tuning
To run a sweep of hyperparameters (Spectral Radius, Leak Rate, Layers, etc.):

```bash
python src/esn_elm/run_deep_esn_tuning.py
```

## 📊 Performance & Experiments

Experiments are logged in `docs/notes.txt`.
*   **Current Best Approach**: DeepESN with ~10-15 layers, low spectral radius, and block-based sequence processing.
*   **Challenges**: Severe class imbalance (dominated by "Free Flowing") and variable sequence lengths.
*   **Metrics**: We optimize for **Macro F1 Score** to ensure performance across all congestion levels, strict adherence to the no-backprop rule.

## 📝 License
This project is licensed under the Apache 2.0 License.
