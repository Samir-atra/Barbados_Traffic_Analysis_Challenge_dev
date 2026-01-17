"""Keras Tuner hyperparameter optimization for Forward-Forward traffic prediction model.

This module implements hyperparameter tuning using keras-tuner for the FF model
defined in ff_restructured.py. It uses RandomSearch or BayesianOptimization to 
search through hyperparameter space for optimal model configuration.

Workflow:
1. Load and preprocess traffic data from Train.csv (same as ff_restructured.py).
2. Define hyperparameter search space.
3. Run RandomSearch/BayesianOptimization to find optimal hyperparameters.
4. Display and save best hyperparameters.
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
import keras_tuner as kt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import random

# Import model classes and data processing from ff_restructured
# Add parent directory to path to import from sibling module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ff_restructured import (
    FFNetwork, FFDense, MacroPrecision, MacroRecall,
    create_dataset_splits, plot_training_results, SEED
)

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
keras.utils.set_random_seed(SEED)


class FFHyperModel(kt.HyperModel):
    """HyperModel for Forward-Forward network hyperparameter tuning.
    
    Defines the hyperparameter search space and model building logic
    for the FF traffic prediction model.
    """
    
    def __init__(self, input_dim, global_epochs=30, batch_size=64):
        """Initialize the HyperModel.
        
        Args:
            input_dim: Input dimension (feature length including label padding).
            global_epochs: Number of global training epochs (fixed).
            batch_size: Batch size for training (fixed).
        """
        self.input_dim = input_dim
        self.global_epochs = global_epochs
        self.batch_size = batch_size
    
    def build(self, hp):
        """Build a model with hyperparameters from the search space.
        
        Args:
            hp: HyperParameters object from keras-tuner.
            
        Returns:
            Compiled FFNetwork model ready for training.
        """
        # Architecture hyperparameters
        num_layers = hp.Int('num_layers', min_value=3, max_value=12, step=1)
        
        # Units per layer - can choose uniform or varying
        use_uniform = hp.Boolean('use_uniform_units', default=True)
        if use_uniform:
            units = hp.Choice('units', values=[32, 64, 128, 256, 512])
            dims = [self.input_dim] + [units] * num_layers
        else:
            # Varying units - first layer, middle layers, last layer
            first_units = hp.Choice('first_units', values=[64, 128, 256])
            middle_units = hp.Choice('middle_units', values=[32, 64, 128])
            last_units = hp.Choice('last_units', values=[16, 32, 64])
            
            dims = [self.input_dim, first_units]
            if num_layers > 2:
                dims.extend([middle_units] * (num_layers - 2))
            if num_layers > 1:
                dims[-1] = last_units
        
        # Learning rate and schedule
        initial_lr = hp.Float('initial_learning_rate', min_value=1e-5, max_value=1e-2, 
                              sampling='log')
        
        # Calculate total training steps for learning rate schedule
        # (This will be approximate, actual steps depend on dataset size)
        local_layer_epochs = hp.Int('layer_epochs', min_value=20, max_value=100, step=5)
        # Estimate: assume ~100 batches per epoch (will be recalculated in fit)
        estimated_batches_per_epoch = 100
        total_train_steps = self.global_epochs * estimated_batches_per_epoch * local_layer_epochs
        
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=total_train_steps,
            alpha=0.1
        )
        
        # Forward-Forward specific hyperparameters
        threshold = hp.Float('threshold', min_value=0.5, max_value=4.0, step=0.1)
        gamma = hp.Float('gamma', min_value=0.5, max_value=5.0, step=0.1)
        
        # Regularization
        l2_reg = hp.Float('l2_reg', min_value=1e-6, max_value=1e-2, sampling='log')
        reg = keras.regularizers.L2(l2_reg)
        
        # EMA settings
        use_ema = hp.Boolean('use_ema', default=True)
        if use_ema:
            ema_overwrite_frequency = hp.Int('ema_overwrite_frequency', 
                                            min_value=1, max_value=10, step=1)
        else:
            ema_overwrite_frequency = None
        
        # Build model
        model = FFNetwork(
            dims=dims,
            kernel_regularizer=reg,
            learning_rate=lr_schedule,
            use_ema=use_ema,
            ema_overwrite_frequency=ema_overwrite_frequency,
            layer_epochs=local_layer_epochs,
            threshold=threshold,
            gamma=gamma
        )
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, global_clipnorm=1.0),
            jit_compile=hp.Boolean('jit_compile', default=False)
        )
        
        return model


def main():
    """Main execution block for hyperparameter tuning."""
    base = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev"
    train_path = os.path.join(base, "demos/Train.csv")
    
    # Output directories
    tuner_dir = os.path.join(base, "keras_tuner_search")
    project_name = "ff_restructured"
    os.makedirs(tuner_dir, exist_ok=True)
    
    # Choose tuner type: 'random' or 'bayesian'
    tuner_type = 'random'  # Change to 'bayesian' for Bayesian Optimization
    
    print("=" * 60)
    print("Forward-Forward Hyperparameter Tuning")
    print(f"Using {tuner_type.upper()} Search")
    print("=" * 60)
    
    # Load and prepare data (same as ff_restructured.py)
    print("\nPreparing Data...")
    X_train, y_train, X_val, y_val = create_dataset_splits(train_path, val_split=0.2)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Input dimension: {X_train.shape[1]}")
    
    # Create datasets
    batch_size = 64
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
    
    # Compute class weights for balanced training
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weight_dict = dict(zip(classes, weights))
    print(f"\nClass weights: {class_weight_dict}")
    
    # Create HyperModel
    input_dim = X_train.shape[1]
    hypermodel = FFHyperModel(
        input_dim=input_dim,
        global_epochs=30,  # Fixed for tuning
        batch_size=batch_size
    )
    
    # Initialize tuner based on selected type
    if tuner_type.lower() == 'random':
        tuner = kt.RandomSearch(
            hypermodel,
            objective=kt.Objective("val_f1", direction="max"),
            max_trials=50,  # Number of trials to run
            executions_per_trial=1,  # Number of models to train per trial
            directory=tuner_dir,
            project_name=project_name,
            seed=SEED,
            overwrite=False  # Set to True to restart search from scratch
        )
    elif tuner_type.lower() == 'bayesian':
        tuner = kt.BayesianOptimization(
            hypermodel,
            objective=kt.Objective("val_f1", direction="max"),
            max_trials=50,  # Number of trials to run
            executions_per_trial=1,  # Number of models to train per trial
            directory=tuner_dir,
            project_name=project_name,
            seed=SEED,
            overwrite=False  # Set to True to restart search from scratch
        )
    else:
        raise ValueError(f"Unknown tuner type: {tuner_type}. Use 'random' or 'bayesian'")
    
    # Display search space summary
    print("\n" + "=" * 60)
    print("Hyperparameter Search Space Summary")
    print("=" * 60)
    print("\nArchitecture:")
    print("  - num_layers: 3-12")
    print("  - units: 32, 64, 128, 256, 512 (uniform) OR varying")
    print("\nTraining:")
    print("  - initial_learning_rate: 1e-5 to 1e-2 (log scale)")
    print("  - layer_epochs: 20-100 (step 5)")
    print("\nForward-Forward Specific:")
    print("  - threshold: 0.5-4.0 (step 0.1)")
    print("  - gamma: 0.5-5.0 (step 0.1)")
    print("\nRegularization:")
    print("  - l2_reg: 1e-6 to 1e-2 (log scale)")
    print("  - use_ema: True/False")
    print("  - ema_overwrite_frequency: 1-10 (if EMA enabled)")
    print("\nOptimization:")
    print("  - jit_compile: True/False")
    print(f"\nTuner Configuration:")
    print(f"  - Type: {tuner_type.upper()}")
    print(f"  - Max trials: {tuner.max_trials}")
    print(f"  - Executions per trial: {tuner.executions_per_trial}")
    print("=" * 60)
    
    # Optional: Resume from previous search
    try:
        tuner.reload()
        print("\nResumed previous search.")
        print(f"Number of completed trials: {len(tuner.oracle.get_trials())}")
    except Exception as e:
        print(f"\nStarting new search. ({e})")
    
    # Run hyperparameter search
    print("\n" + "=" * 60)
    print(f"Starting {tuner_type.upper()} Search")
    print("=" * 60)
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_f1',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    tuner.search(
        train_dataset,
        validation_data=val_dataset,
        epochs=30,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=2
    )
    
    # Get best hyperparameters
    print("\n" + "=" * 60)
    print("Best Hyperparameters Found")
    print("=" * 60)
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("\nArchitecture:")
    print(f"  num_layers: {best_hps.get('num_layers')}")
    if best_hps.get('use_uniform_units'):
        print(f"  units (uniform): {best_hps.get('units')}")
    else:
        print(f"  first_units: {best_hps.get('first_units')}")
        print(f"  middle_units: {best_hps.get('middle_units')}")
        print(f"  last_units: {best_hps.get('last_units')}")
    
    print("\nTraining:")
    print(f"  initial_learning_rate: {best_hps.get('initial_learning_rate')}")
    print(f"  layer_epochs: {best_hps.get('layer_epochs')}")
    
    print("\nForward-Forward:")
    print(f"  threshold: {best_hps.get('threshold')}")
    print(f"  gamma: {best_hps.get('gamma')}")
    
    print("\nRegularization:")
    print(f"  l2_reg: {best_hps.get('l2_reg')}")
    print(f"  use_ema: {best_hps.get('use_ema')}")
    if best_hps.get('use_ema'):
        print(f"  ema_overwrite_frequency: {best_hps.get('ema_overwrite_frequency')}")
    
    print("\nOptimization:")
    print(f"  jit_compile: {best_hps.get('jit_compile')}")
    
    # Get best model and evaluate
    print("\n" + "=" * 60)
    print("Evaluating Best Model")
    print("=" * 60)
    
    best_model = tuner.get_best_models(num_models=1)[0]
    
    print("\nBest model summary:")
    best_model.summary()
    
    # Evaluate on validation set
    val_results = best_model.evaluate(val_dataset, verbose=1)
    print("\nValidation Results:")
    metric_names = best_model.metrics_names
    for name, value in zip(metric_names, val_results):
        print(f"  {name}: {value:.4f}")
    
    # Save best hyperparameters to file
    results_file = os.path.join(tuner_dir, f"{project_name}_best_hyperparameters.txt")
    with open(results_file, 'w') as f:
        f.write(f"Best Hyperparameters from {tuner_type.upper()} Search\n")
        f.write("=" * 60 + "\n\n")
        for hp_name in best_hps.values:
            f.write(f"{hp_name}: {best_hps.get(hp_name)}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("Validation Results:\n")
        for name, value in zip(metric_names, val_results):
            f.write(f"  {name}: {value:.4f}\n")
    
    print(f"\nBest hyperparameters saved to: {results_file}")
    
    # Save best model
    model_path = os.path.join(tuner_dir, f"{project_name}_best_model.keras")
    best_model.save(model_path)
    print(f"Best model saved to: {model_path}")
    
    print("\n" + "=" * 60)
    print("Hyperparameter Tuning Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()