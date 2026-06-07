# FF Hyperband TPU Configuration Summary

## Overview
Created `ff_hyperband_tpu_kernel.py` - a TPU-optimized version of the Forward-Forward Hyperband hyperparameter search script for Kaggle.

## Key Changes from GPU Version

### 1. **Backend Configuration**
- **GPU Version**: Uses TensorFlow backend (`os.environ["KERAS_BACKEND"] = "tensorflow"`)
- **TPU Version**: Uses JAX backend (`os.environ["KERAS_BACKEND"] = "jax"`)
- **Reason**: JAX provides better TPU support and performance on Kaggle

### 2. **TPU Initialization**
```python
# Added TPU detection and initialization
try:
    tpu = jax.devices('tpu')[0]
    print(f"TPU devices found: {jax.devices()}")
    print(f"TPU device count: {jax.device_count()}")
except:
    print("No TPU found, using CPU/GPU")
```

### 3. **Random Number Generation**
- **GPU Version**: Uses `tf.random.uniform()` for generating negative samples
- **TPU Version**: Uses `jax.random.randint()` with PRNGKey for JAX compatibility
```python
rng = random.PRNGKey(SEED)
offsets = random.randint(rng, shape=[batch_size], minval=1, maxval=4)
```

### 4. **Dynamic Slice Operations**
- **GPU Version**: Uses `xla.dynamic_update_slice()` for label overlay
- **TPU Version**: Uses standard JAX operations with `ops.one_hot()` and `ops.concatenate()`
```python
# Simplified overlay operation for TPU
update = ops.one_hot(y_idx, 4) * 10.0
x_updated = ops.concatenate([update, x[4:]], axis=0)
```

### 5. **Batch Size Optimization**
- **GPU Version**: Batch size = 64
- **TPU Version**: Batch size = 128 (larger for TPU efficiency)
- **Reason**: TPUs perform better with larger batch sizes

### 6. **Training Configuration**
- **GPU Version**: 15 global epochs
- **TPU Version**: 10 global epochs (reduced for faster tuning)
- Both use the same Hyperband configuration with factor=3

### 7. **Data Handling**
- **GPU Version**: Uses `tf.data.Dataset` with shuffling and prefetching
- **TPU Version**: Passes NumPy arrays directly to `fit()` method
```python
# TPU version uses direct NumPy arrays
best_model.fit(
    X_train_np,
    y_train_np,
    validation_data=(X_val_np, y_val_np),
    ...
)
```

### 8. **Documentation**
- Added comprehensive Google-style docstrings to all functions and classes
- Improved code comments for better readability

## Hyperparameter Search Space

Both versions search the same hyperparameter space:
- **num_layers**: 4-16 layers
- **units**: 16-256 units per layer (step=16)
- **learning_rate**: 1e-4 to 1e-2 (log scale)
- **layer_epochs**: 20-80 (step=5)
- **threshold**: 0.5-4.0 (step=0.1)
- **gamma**: 1.0-5.0 (step=0.1)
- **l2_reg**: 1e-6 to 1e-3 (log scale)
- **ema_overwrite_frequency**: 1-10

## Output Files

The TPU version saves:
1. **best_ff_tpu_weights.weights.h5** - Best model weights
2. **best_hyperparameters.txt** - Detailed hyperparameter values

## Kaggle Metadata

Created `kernel-metadata-hyperband-tpu.json`:
- Kernel ID: `samerattrah/barbados-traffic-ff-hyperband-tpu`
- TPU enabled: `true`
- GPU enabled: `false`
- Internet enabled: `true` (for package installations)

## Usage on Kaggle

1. Upload both files to Kaggle:
   - `ff_hyperband_tpu_kernel.py`
   - `kernel-metadata-hyperband-tpu.json`

2. Ensure dataset is attached:
   - `samerattrah/barbados-traffic-analysis-challenge`

3. Run the kernel with TPU v5e-8 accelerator

## Expected Performance

- **TPU Advantages**:
  - Faster matrix operations
  - Better handling of large batch sizes
  - Efficient parallel processing across TPU cores
  
- **Considerations**:
  - TPU v5e-8 on Kaggle has 16GB HBM
  - Batch size of 128 should fit comfortably
  - Hyperband search will explore ~30-50 configurations

## Notes

- The script maintains the same Forward-Forward algorithm logic
- All metrics (accuracy, F1, precision, recall) are preserved
- Class weighting is still applied for imbalanced data
- Cosine decay learning rate schedule is maintained
