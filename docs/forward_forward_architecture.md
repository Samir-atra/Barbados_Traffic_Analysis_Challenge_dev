# Forward-Forward Algorithm for Traffic Prediction: Technical Documentation

This document provides a comprehensive breakdown of the Forward-Forward (FF) implementation for the Barbados Traffic Analysis Challenge. The algorithm replaces traditional backpropagation with a layer-wise local learning rule inspired by Geoffrey Hinton's Forward-Forward algorithm.

## 1. Core Principles: The Forward-Forward Idea

Unlike standard neural networks that use a forward pass to get a scalar loss and a backward pass to compute gradients, the FF algorithm uses **two forward passes**:

1.  **Positive Pass**: The network is shown "real" data (correct features paired with the correct label). The goal is to maximize the "goodness" (sum of squared activations) for these samples.
2.  **Negative Pass**: The network is shown "fake" data (correct features paired with an incorrect label). The goal is to minimize the "goodness" for these samples.

Training happens **locally at each layer**. Each layer learns to distinguish between positive and negative signals independently of the layers above it.

---

## 2. Data Representation & Preprocessing

The script `ff_restructured.py` transforms raw traffic data into a format suitable for the FF algorithm.

### Feature Engineering (`get_features_and_labels`)
*   **Temporal Features**: `hour`, `minute`, and `day_of_week` are normalized (scaled between 0 and 1).
*   **Spatial Features**: `view_label` (the 4 camera locations) is converted to a 4-dimensional one-hot vector and a `view_id`.
*   **Segment ID**: The `time_segment_id` is normalized by 5000.
*   **Joint Labeling**: The competition requires predicting Enter and Exit congestion ratings (4 classes each). These are combined into a single **16-class joint label** ($4 \times 4$).

### Sequential Windowing (`create_sequential_dataset`)
The model is trained to predict the state at $T+1$ given features at $T$. Data is grouped by camera view and sorted by time. Only continuous blocks (no gaps in segment IDs) are used.

### The Label Overlay (`overlay_y_on_x`)
To teach the network which label is correct, we "overlay" the label onto the input features.
*   The first 16 dimensions of the input vector are reserved for a 1-hot representation of the label.
*   A sample for label `j` will have a value of `1.0` at index `j` and `0.0` at the other 15 indices.
*   **Total Feature Vector**: 16 (labels) + 9 (base features) = **25 Dimensions**.

---

## 3. Network Architecture

The network is composed of two primary classes: `FFDense` and `FFNetwork`.

### `FFDense` Layer
This is the fundamental unit of the network.
*   **Input Normalization**: Before every pass, the input is normalized by its L2 norm (`x / ||x||_2`). This ensures the layer can only use the **direction** of the vector, preventing it from simply increasing weight magnitudes to achieve higher "goodness."
*   **Threshold**: A constant (default `1.5`) used to separate positive and negative goodness.
*   **Forward-Forward Loop**: For every global epoch, each layer runs internal epochs (default `60`) to optimize its weights against the local loss:
    $$Loss = \log(1 + e^{[-G_{pos} + \theta, G_{neg} - \theta]})$$
    where $G$ is the mean squared activation (Goodness).

### `FFNetwork` Model
This class orchestrates the layers and handles the global training loop.
*   **Dimensions**: Currently configured as `[25, 512, 512, 512, 512]` (Input + 4 Hidden Layers).
*   **Loss Tracking**: Aggregates the local losses from all layers to provide a global training indicator.
*   **Metrics**: Implements `SparseCategoricalAccuracy` and `F1Score` (Macro) to evaluate performance during training and validation.

---

## 4. Training Procedure (`train_step`)

For each batch:
1.  **Generate Samples**:
    *   **Positive**: Real features paired with their true labels.
    *   **Negative**: Real features paired with labels shuffled randomly within the batch.
2.  **Sequential Training**:
    *   The first layer trains on the input samples.
    *   Once trained, the outputs of the first layer (normalized) serve as the inputs for the second layer.
    *   This continues through all layers.
3.  **No Backprop**: Gradients are calculated locally via `tf.GradientTape` within the `FFDense.forward_forward` method.

---

## 5. Prediction Logic (`predict_one`)

Because the network doesn't have a Softmax output layer for all classes, inference requires testing all possibilities:
1.  Take an input feature vector.
2.  For each of the **16 possible labels**:
    *   Overlay the label on the feature vector.
    *   Pass the vector through the entire network.
    *   Calculate the **total goodness** (sum of squared activations across all layers).
3.  Select the label that produced the **highest total goodness**.

---

## 6. Autoregressive Inference for Competition

For the Test set, we need to predict 5 steps into the future ($T+3$ to $T+7$):
1.  **Seed State**: Start with the last known features for a specific camera view.
2.  **Step-by-Step Prediction**:
    *   Predict the state at $T+1$.
    *   Update the feature vector (increment `time_segment_id` and clock time).
    *   Use the predicted state as input for the next step (Autoregressive).
3.  **Submission**: Filter and format the predictions for $T+3$ through $T+7$ to match the `SampleSubmission.csv` requirements.

---

## 7. Hyperparameters Summary

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `dims` | `[25, 512, 512, 512, 512]` | 1 Input + 4 Hidden Layers |
| `Layer Units` | `512` | Neurons per hidden layer |
| `Layer Epochs` | `60` | Internal optimization loops per layer |
| `FF Learning Rate`| `0.003` | Adam optimizer for each layer |
| `Threshold` | `1.5` | Margin for the goodness function |
| `Global Epochs` | `500` | Total training iterations |
| `Batch Size` | `2048` | Number of sequences processed at once |
| `Optimizer` | `Adam` | Used for both local and global coordination |
