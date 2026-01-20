"""Data Processing Utilities for Traffic Prediction.

This module provides shared data processing functions for traffic prediction
models. It includes the fixed-length block chunking strategy where:
- Blocks are chunked into segments of maximum CHUNK_SIZE (100) elements
- Shorter blocks or final chunks are padded to exactly CHUNK_SIZE
- This ensures consistent input dimensions across all training samples

Example:
    - A block with 46 elements → 1 chunk padded to 100
    - A block with 645 elements → 7 chunks (6 full + 1 padded to 100)
"""

import os
import numpy as np
import pandas as pd


# Configuration
CHUNK_SIZE = 100  # Maximum elements per chunk
SEQ_LEN = 15      # Sequence length for sliding windows (original resolution)
DOWNSAMPLE_FACTOR = 1  # No downsampling (kept for compatibility)


def get_features_and_labels(df):
    """Extracts features and labels with enhanced feature engineering.
    
    Args:
        df: A pandas DataFrame containing raw traffic data.
        
    Returns:
        A tuple (features, labels):
            - features: A float32 NumPy array of shape (N, 20).
            - labels: An int32 NumPy array of shape (N,) or None.
    """
    df = df.copy()
    df['video_time'] = pd.to_datetime(df['video_time'])
    
    hour = df['video_time'].dt.hour
    minute = df['video_time'].dt.minute
    day_of_week = pd.to_datetime(df['date']).dt.dayofweek
    
    # Cyclical encoding removed
    
    # Peak hour indicators removed
    
    # Linear time features
    df['hour_norm'] = hour / 23.0
    df['minute_norm'] = minute / 59.0
    
    view_map = {
        'Norman Niles #1': 0, 'Norman Niles #2': 1, 
        'Norman Niles #3': 2, 'Norman Niles #4': 3
    }
    df['view_id'] = df['view_label'].map(view_map)
    df['seg_id_norm'] = df['time_segment_id'] / 5000.0
    
    congestion_map = {
        'free flowing': 0, 'light delay': 1, 
        'moderate delay': 2, 'heavy delay': 3
    }
    if 'congestion_enter_rating' in df.columns:
        df['enter_id'] = df['congestion_enter_rating'].map(
            congestion_map).fillna(0).astype(int)
        labels = df['enter_id'].values
    else:
        labels = None
    
    view_1hot = pd.get_dummies(df['view_id'], prefix='view').reindex(
        columns=['view_0', 'view_1', 'view_2', 'view_3'], 
        fill_value=0).astype(float).values
    
    if 'signaling' in df.columns:
        sig_map = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
        df['sig_id'] = df['signaling'].map(sig_map).fillna(0)
    else:
        df['sig_id'] = 0
    
    sig_1hot = pd.get_dummies(df['sig_id'], prefix='sig').reindex(
        columns=['sig_0', 'sig_1', 'sig_2', 'sig_3'], 
        fill_value=0).astype(float).values
    
    features = np.concatenate([
        df[['hour_norm', 'minute_norm', 'seg_id_norm']].values,
        (df[['view_id']].values / 3.0),
        view_1hot,
        sig_1hot
    ], axis=1).astype('float32')

    return features, labels


def identify_blocks(group):
    """Identifies continuous sequential blocks within a data group.
    
    Args:
        group: A pandas DataFrame with a 'time_segment_id' column.
        
    Returns:
        The group with an added 'block_id' column.
    """
    group = group.sort_values('time_segment_id')
    ids = group['time_segment_id'].values
    is_break = np.zeros(len(ids), dtype=int)
    is_break[1:] = (ids[1:] != ids[:-1] + 1).astype(int)
    group['block_id'] = np.cumsum(is_break)
    return group


def chunk_and_pad_block(features, labels, chunk_size=CHUNK_SIZE):
    """Chunks a block into fixed-size segments with padding.
    
    Args:
        features: NumPy array of shape (N, F) where N is block length.
        labels: NumPy array of shape (N,) or None.
        chunk_size: Maximum elements per chunk (default: 100).
        
    Returns:
        A list of tuples (chunk_features, chunk_labels) where each chunk
        has exactly chunk_size elements.
    """
    n_elements = len(features)
    n_features = features.shape[1]
    
    if n_elements == 0:
        return []
    
    # Calculate number of chunks needed
    n_chunks = (n_elements + chunk_size - 1) // chunk_size
    
    chunks = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, n_elements)
        
        chunk_feats = features[start_idx:end_idx]
        chunk_labs = labels[start_idx:end_idx] if labels is not None else None
        
        # Pad if necessary
        actual_len = len(chunk_feats)
        if actual_len < chunk_size:
            # Pad features by repeating the last element
            pad_count = chunk_size - actual_len
            feat_padding = np.tile(chunk_feats[-1:], (pad_count, 1))
            chunk_feats = np.vstack([chunk_feats, feat_padding])
            
            if chunk_labs is not None:
                # Pad labels by repeating the last label
                lab_padding = np.full(pad_count, chunk_labs[-1])
                chunk_labs = np.concatenate([chunk_labs, lab_padding])
        
        chunks.append((chunk_feats, chunk_labs, actual_len))
    
    return chunks


def create_sequences_from_chunk(features, labels, seq_len=SEQ_LEN, 
                                 actual_len=None):
    """Creates sliding window sequences from a chunk.
    
    Args:
        features: NumPy array of shape (chunk_size, F).
        labels: NumPy array of shape (chunk_size,) or None.
        seq_len: Sequence length for sliding windows.
        actual_len: Actual length before padding (for masking).
        
    Returns:
        Tuple (X_sequences, y_labels) where:
            - X_sequences: List of (seq_len, F) arrays
            - y_labels: List of target labels
    """
    if actual_len is None:
        actual_len = len(features)
    
    X_seqs = []
    y_labs = []
    
    # Only create windows up to the actual data (not padded region)
    for i in range(actual_len - seq_len):
        X_seqs.append(features[i:i + seq_len])
        if labels is not None:
            y_labs.append(labels[i + seq_len])
    
    return X_seqs, y_labs


def create_raw_sequences_chunked(csv_path, val_split=0.2, seq_len=SEQ_LEN,
                                  chunk_size=CHUNK_SIZE):
    """Processes CSV into fixed-size chunked sequences WITHOUT data loss.
    
    This function:
    1. Groups data by view_label
    2. Identifies continuous blocks within each view
    3. Splits each block 80/20 FIRST (train/val on block level)
    4. THEN chunks each portion into fixed segments
    5. Creates sliding window sequences from each chunk
    
    This approach preserves more data by doing the split before chunking.
    
    Args:
        csv_path: Path to the CSV file (Train.csv).
        val_split: Fraction of data to use for validation.
        seq_len: Sequence length for sliding windows.
        chunk_size: Maximum elements per chunk.
        
    Returns:
        Tuple (X_train, y_train, X_val, y_val) of NumPy arrays.
    """
    df = pd.read_csv(csv_path)
    train_X, train_y = [], []
    val_X, val_y = [], []
    
    total_chunks = 0
    padded_elements = 0
    
    print(f"Loading {len(df)} rows from {csv_path}...")
    print(f"Chunk size: {chunk_size}, Sequence length: {seq_len}")
    
    # 1. Collect all valid blocks across all views
    all_blocks_data = []
    for view_label, view_group in df.groupby('view_label'):
        view_group = identify_blocks(view_group)
        for block_id, block in view_group.groupby('block_id'):
            if len(block) >= (SEQ_LEN + 1):
                all_blocks_data.append(block)
    
    # 2. Shuffle blocks globally
    # Using a fixed seed for reproducibility across runs
    import random
    rng = random.Random(42)
    rng.shuffle(all_blocks_data)
    
    # 3. Split blocks into Train and Val sets (Block-level split)
    n_blocks = len(all_blocks_data)
    n_train_blocks = int(n_blocks * (1 - val_split))
    
    train_blocks = all_blocks_data[:n_train_blocks]
    val_blocks = all_blocks_data[n_train_blocks:]
    
    def process_block_list(blocks, is_train=True):
        X_out, y_out = [], []
        chunks_count = 0
        padded_count = 0
        
        for block in blocks:
            # Extract features/labels (no downsampling)
            features, labels = get_features_and_labels(block)
            
            # Chunk the features directly
            chunks = chunk_and_pad_block(features, labels, chunk_size)
            chunks_count += len(chunks)
            
            for chunk_feats, chunk_labs, actual_len in chunks:
                padded_count += chunk_size - actual_len
                if actual_len >= seq_len + 1:
                    seqs, seq_labs = create_sequences_from_chunk(
                        chunk_feats, chunk_labs, seq_len, actual_len)
                    X_out.extend(seqs)
                    y_out.extend(seq_labs)
        return X_out, y_out, chunks_count, padded_count

    print(f"Distributing {n_blocks} blocks: {n_train_blocks} Train, {n_blocks - n_train_blocks} Val")
    
    train_X, train_y, t_chunks, t_padded = process_block_list(train_blocks)
    val_X, val_y, v_chunks, v_padded = process_block_list(val_blocks)
    
    total_chunks = t_chunks + v_chunks
    padded_elements = t_padded + v_padded
    
    train_y_np = np.array(train_y)
    val_y_np = np.array(val_y)
    
    t_cls, t_cnt = np.unique(train_y_np, return_counts=True)
    v_cls, v_cnt = np.unique(val_y_np, return_counts=True)
    
    print(f"\nChunking Statistics:")
    print(f"  Total chunks created: {total_chunks}")
    print(f"  Total padded elements: {padded_elements}")
    print(f"\nLoaded {len(train_y_np)} train samples: {dict(zip(t_cls, t_cnt))}")
    print(f"Loaded {len(val_y_np)} val samples: {dict(zip(v_cls, v_cnt))}")
    
    X_train_np = np.array(train_X)
    X_val_np = np.array(val_X)
    
    # Final check: Ensure features are strictly within [0, 1]
    # We clip here to handle any potential floating point artifacts or 
    # slightly out-of-range segment IDs if they exceed 5000
    X_train_np = np.clip(X_train_np, 0.0, 1.0)
    X_val_np = np.clip(X_val_np, 0.0, 1.0)
    
    return X_train_np, train_y_np, X_val_np, val_y_np



def analyze_chunked_blocks(csv_path, chunk_size=CHUNK_SIZE):
    """Analyzes how blocks will be chunked.
    
    Args:
        csv_path: Path to the CSV file.
        chunk_size: Maximum elements per chunk.
        
    Returns:
        DataFrame with chunk analysis.
    """
    df = pd.read_csv(csv_path)
    
    analysis = []
    
    for view_label, view_group in df.groupby('view_label'):
        view_group = identify_blocks(view_group)
        
        for block_id, block in view_group.groupby('block_id'):
            block_len = len(block) // DOWNSAMPLE_FACTOR
            if block_len < 1:
                continue
                
            n_chunks = (block_len + chunk_size - 1) // chunk_size
            last_chunk_len = block_len % chunk_size
            if last_chunk_len == 0:
                last_chunk_len = chunk_size
            padding_needed = chunk_size - last_chunk_len
            
            analysis.append({
                'view_label': view_label,
                'block_id': block_id,
                'original_length': len(block),
                'downsampled_length': block_len,
                'n_chunks': n_chunks,
                'last_chunk_actual_len': last_chunk_len,
                'padding_needed': padding_needed,
                'total_padded_length': n_chunks * chunk_size
            })
    
    return pd.DataFrame(analysis)


if __name__ == "__main__":
    # Demo usage
    base = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev"
    train_path = os.path.join(base, "demos/Train.csv")
    
    if os.path.exists(train_path):
        print("=" * 60)
        print("CHUNK ANALYSIS")
        print("=" * 60)
        
        analysis_df = analyze_chunked_blocks(train_path)
        print(f"\nTotal original blocks: {len(analysis_df)}")
        print(f"Total chunks after splitting: {analysis_df['n_chunks'].sum()}")
        print(f"Total padding elements: {analysis_df['padding_needed'].sum()}")
        
        print("\nChunks per original block distribution:")
        print(analysis_df['n_chunks'].value_counts().sort_index())
        
        print("\n" + "=" * 60)
        print("SEQUENCE CREATION WITH CHUNKING")
        print("=" * 60)
        
        X_train, y_train, X_val, y_val = create_raw_sequences_chunked(train_path)
        print(f"\nFinal shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  X_val:   {X_val.shape}")
        print(f"  y_val:   {y_val.shape}")
