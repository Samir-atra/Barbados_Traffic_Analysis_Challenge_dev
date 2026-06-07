# Block Generation Methodology

This document explains the logic and criteria used to segment the traffic dataset into sequential blocks for training machine learning models.

## Overview

The traffic dataset consists of time-series observations from traffic cameras at the Norman Niles intersection in Barbados. Each row represents a 5-minute time segment containing features like congestion ratings, signaling status, and temporal information.

For sequential prediction models (ESN, PCN, LSTM, etc.), the data must be organized into **contiguous time blocks** where observations are consecutive in time.

## Block Definition

A **block** is defined as a sequence of consecutive `time_segment_id` values from the same camera view (`view_label`).

### Block Breaking Criteria

A new block starts when:
1. **Time gap detected**: The `time_segment_id` is not consecutively incrementing (i.e., `current_id != previous_id + 1`)
2. **View change**: Moving to a different `view_label` (Norman Niles #1, #2, #3, or #4)

### Why Blocks Are Necessary

1.  **Temporal Integrity (Causality)**: In traffic prediction, the road state at 8:05 AM is a direct consequence of what happened at 8:00 AM. If there is a gap (e.g., a 2-hour camera outage), the causal link is broken.
2.  **Hidden State and Memory**: Biological-inspired models like **Predictive Coding Networks (PCN)** and **Echo State Networks (ESN)** build up internal hidden states ("memory") as they process data. Feeding the model non-consecutive segments (e.g., Monday followed by Tuesday) would mix unrelated temporal contexts, resulting in unstable internal states.
3.  **Preventing "Time-Travel" Errors**: Breaking the data ensures the model never learns from "jumps" in time that didn't occur in reality.
4.  **Avoiding Padding Loss**: If we treated whole days as single sequences, we would have to pad shorter days with zeros. Because blocks are broken logically, we only process high-quality, continuous signal.

## Block Generation Algorithm

```python
def identify_blocks(group):
    """
    Identifies continuous sequential blocks within a view group.
    
    Args:
        group: DataFrame with data from a single view_label
        
    Returns:
        DataFrame with 'block_id' column added
    """
    # Sort by time_segment_id to ensure temporal ordering
    group = group.sort_values('time_segment_id')
    
    # Get the time segment IDs
    ids = group['time_segment_id'].values
    
    # Detect breaks: where current_id != previous_id + 1
    is_break = np.zeros(len(ids), dtype=int)
    is_break[1:] = (ids[1:] != ids[:-1] + 1).astype(int)
    
    # Cumulative sum gives unique block IDs
    group['block_id'] = np.cumsum(is_break)
    
    return group
```

## Processing Pipeline

The complete block generation follows this pipeline:

```
Dataset (Train.csv)
    │
    ├── Group by view_label
    │       │
    │       ├── Norman Niles #1 ──┬── Sort by time_segment_id
    │       │                     └── Detect breaks → Assign block_ids
    │       │
    │       ├── Norman Niles #2 ──┬── Sort by time_segment_id
    │       │                     └── Detect breaks → Assign block_ids
    │       │
    │       ├── Norman Niles #3 ──┬── Sort by time_segment_id
    │       │                     └── Detect breaks → Assign block_ids
    │       │
    │       └── Norman Niles #4 ──┬── Sort by time_segment_id
    │                             └── Detect breaks → Assign block_ids
    │
    └── Result: 180 blocks (45 per view)
```

## Example

Consider this sequence of `time_segment_id` values from a single view:

```
time_segment_ids: [100, 101, 102, 105, 106, 107, 108, 200, 201]
                   └─ Block 0 ─┘  └───── Block 1 ─────┘  └Block 2┘
```

- **Block 0**: [100, 101, 102] - 3 consecutive segments
- **Block 1**: [105, 106, 107, 108] - 4 consecutive segments (gap after 102)
- **Block 2**: [200, 201] - 2 consecutive segments (gap after 108)

## Usability Criteria

For training sequential models, a block must have a minimum length to create input sequences:

| Sequence Length | Minimum Block Size | Reason |
|-----------------|-------------------|--------|
| 15 (default) | 16 | Need 15 input steps + 1 target step |
| 10 | 11 | Need 10 input steps + 1 target step |
| 20 | 21 | Need 20 input steps + 1 target step |

### Training Windows from Blocks

For a block of length `N` and sequence length `L`, the number of training windows is:
```
n_windows = N - L
```
Example: A block with 100 segments and sequence length 15 produces 85 training windows.

## Rationale for Sequence Length Selection

The choice of specific sequence lengths (e.g., 15) is driven by three main factors:

### 1. Capturing Temporal Context
A sequence length of 15 segments represents **75 minutes of history** (5 mins/segment). 
- **Tren Analysis**: This is typically sufficient to capture the "trend"—whether congestion is building up toward a rush hour or clearing after one.
- **Relevance**: 5 minutes is too little context; 5 hours might contain irrelevant historical noise.

### 2. Computational Consistency
Most standard machine learning models (ELM, FF, PCN) require a fixed input size. By setting `seq_len=15`, the input dimensions are always constant (e.g., 15 steps × 20 features = 300 inputs), allowing for:
- **Batch Processing**: Efficient utilization of GPU/TPU hardware by processing samples in parallel.
- **No Padding**: Avoids the need for zero-padding which can confuse the model's understanding of traffic density.

### 3. Sliding Window Augmentation
Using a shorter window length than the total block length allows us to use **Sliding Window Augmentation**. This effectively multiplies our training data, as the model learns how to predict the "next state" from dozens of different starting points within a single continuous hour of traffic.

## Block Statistics (Current Dataset)

Based on the analysis of `Train.csv`:

| Metric | Value |
|--------|-------|
| Total Blocks | 180 |
| Blocks per View | 45 |
| Minimum Length | 47 |
| Maximum Length | 308 |
| Mean Length | 89.31 |
| Median Length | 60 |

### Length Distribution

| Range | Count | Percentage |
|-------|-------|------------|
| 31-50 | 84 | 46.7% |
| 51-100 | 36 | 20.0% |
| 101-200 | 52 | 28.9% |
| 201-500 | 8 | 4.4% |

## Implications for Model Training

### 1. Data Splitting Strategy

When splitting data for training/validation, splits should occur within blocks to:
- Prevent data leakage (overlapping windows across train/val)
- Maintain temporal ordering
- Ensure both sets have representative samples

Current approach: Split each block at 80%/20% temporally, then create windows.

### 2. Class Distribution Within Blocks

Each block may have different class distributions:
- 144/180 blocks are dominated by "Free Flowing"
- High class imbalance requires weighted sampling or loss weighting
- Mean of 32 class transitions per block indicates temporal dynamics

### 3. Sequence Padding

For blocks shorter than the sequence length requirement:
- Pad with repeated first observation
- Or exclude from training (current approach for blocks < 16 samples)

## Related Files

- `src/data_processing/analyze_blocks.py` - Script that generates block statistics
- `analytics/block_analysis.csv` - Detailed per-block analysis output
- `src/esn_elm/esn_traffic.py` - Example of block-based data loading
- `src/pcn/pcn_traffic.py` - Another example implementation

## Summary

Blocks are fundamental units for sequential modeling in this traffic prediction task. They ensure:
1. **Temporal validity** - No predictions across time gaps
2. **Data integrity** - Clean, consecutive observations
3. **Model reliability** - Valid input sequences for training

The current dataset provides 180 well-sized blocks (45 per camera view) with all blocks having sufficient length (≥47) for training with sequence length 15.
