# Forward-Forward (FF) Traffic Prediction Algorithm

This document provides a comprehensive technical explanation of the `ff-script.py` script, which implements Geoffrey Hinton's **Forward-Forward Algorithm** to predict traffic congestion for the "Norman Niles #1" location in Barbados.

## 1. Overall Functionality
The script aims to predict the **future 5 congestion states** (for both entrance and exit rates) based on current temporal and traffic data. Unlike traditional neural networks that use backpropagation, this implementation uses the **Forward-Forward Algorithm**. 

The process follows two main stages:
1.  **Representation Learning (Self-supervised)**: The network learns the "structure" of real traffic data by trying to distinguish it from "negative" (scrambled) data.
2.  **Supervised Prediction**: A linear regression head is trained on the fixed representations learned in the first stage to map them to actual future congestion values.

---

## 2. Data Preprocessing (`load_and_preprocess_data`)

This function prepares the raw `Train.csv` data for the FF model.

### Key Variables:
*   `df`: The pandas DataFrame filtered for `view_label == 'Norman Niles #1'`.
*   `congestion_map`: A dictionary mapping text ratings (e.g., "free flowing") to integers (1-4).
*   **Normalized Features**:
    *   `hour`: Current hour divided by 23 (Range [0, 1]).
    *   `minute`: Current minute divided by 59 (Range [0, 1]).
    *   `day`: Day of the week divided by 6 (Range [0, 1]).
    *   `enter_norm` / `exit_norm`: Congestion levels (1-4) divided by 4.
*   `X`: The input feature vector `[hour, minute, day, enter_norm, exit_norm]`.
*   `y`: Target vector containing 10 values (the enter/exit pairs for the next 5 time intervals).

---

## 3. Layer Architecture: `FFDense`

Each `FFDense` layer is a self-contained unit that learns locally.

### Functionality:
- **Goodness Measure**: Instead of minimizing an error signal from the output, the layer calculates its "Goodness" for a given input. Goodness is defined as the **mean square of the activations** ($activity^2$).
- **Local Objective**: The layer adjusts its internal weights (`self.dense`) to:
    1.  Maximize goodness for **Positive Data** (real sequences).
    2.  Minimize goodness for **Negative Data** (sequences with shuffled features).

### Parameters:
*   `units`: The number of neurons in the layer (default: 512).
*   `threshold`: The target value for goodness. It acts as a boundary to separate positive and negative signals.
*   `num_epochs`: The number of local iterations the layer performs to update its weights.

---

## 4. Model Architecture: `FFNetwork`

The network manages the sequential training of all `FFDense` layers and the final predictor.

### Core Methods:
*   **`train_reps(x_pos, x_neg)`**:
    - Coordinates the layer-wise training. 
    - The first layer trains on the raw input. 
    - The subsequent layers train on the *activations* produced by the previous layer.
*   **`get_activations(x)`**:
    - Passes the input through the trained FF layers.
    - Concatenates the features from all layers into a single high-dimensional vector.
*   **`train_predictor_head(X, y)`**:
    - Once the FF layers are "frozen," this step trains a small MLP (`self.predictor`) to translate the learned activations into the predicted future congestion rates.

---

## 5. Summary of Workflow

1.  **Preparation**: Generate **Positive Data** (real samples) and **Negative Data** (samples where features like Time and Congestion are mismatched).
2.  **Self-supervised Training**: Each `FFDense` layer learns to produce high activity for real traffic patterns and low activity for corrupted ones.
3.  **Feature Extraction**: Convert all training samples into high-level feature vectors using the trained FF layers.
4.  **Regression**: Train the final output layer to predict:
    - `[Enter(t+1), Exit(t+1), Enter(t+2), ..., Exit(t+5)]`.
5.  **Inference**: Given a single 1-minute window of traffic data, the model predicts the trend for the next 5 minutes.
