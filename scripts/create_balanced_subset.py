
import pandas as pd
import numpy as np
import os

def create_balanced_subset(input_csv, output_csv, n_total=3000, block_size=50):
    """
    Creates a balanced subset of the training data.
    
    Criteria:
    1. Total dataset size: ~n_total
    2. Balanced across 4 congestion ratings (as much as possible).
    3. Balanced across 4 view labels (Norman Niles #1, #2, #3, #4).
    4. Data must be in sequential blocks of `block_size` videos.
    """
    
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    df = pd.read_csv(input_csv)
    print(f"Original Dataset: {len(df)} samples")
    
    # Ensure sorted by view and time for sequential blocking
    if 'datetimestamp_start' in df.columns:
        df['dt'] = pd.to_datetime(df['datetimestamp_start'])
        df = df.sort_values(by=['view_label', 'dt'])
    else:
        df = df.sort_values(by=['view_label', 'time_segment_id'])
        
    # Create Blocks
    # A block is defined as a contiguous chunk of `block_size` rows within a view
    # We assign a unique block_id to every chunk
    
    df['temp_block_id'] = -1
    block_counter = 0
    
    # We only want full blocks (or at least close to full) to maintain strict structure
    final_blocks = []
    
    for view in df['view_label'].unique():
        view_df = df[df['view_label'] == view]
        
        # Calculate how many full blocks fit
        n_rows = len(view_df)
        n_blocks = n_rows // block_size
        
        if n_blocks == 0: continue
            
        # Truncate to multiple of block_size to ensure clean blocks
        truncated_view_df = view_df.iloc[:n_blocks * block_size].copy()
        
        # Assign IDs
        # Array of [0, 0, ..., 1, 1, ..., N, N]
        ids = np.repeat(np.arange(block_counter, block_counter + n_blocks), block_size)
        truncated_view_df['temp_block_id'] = ids
        
        block_counter += n_blocks
        final_blocks.append(truncated_view_df)
        
    df_blocked = pd.concat(final_blocks)
    
    # Analyze Blocks to Select the Best Subset
    # We need to select N blocks such that N * block_size ~= n_total
    target_blocks = n_total // block_size
    print(f"Targeting {target_blocks} blocks of size {block_size} (Total: {target_blocks*block_size})")
    
    # We need to balance across Views and Labels.
    # Since a block contains mixed labels, we characterize each block by its 'dominant' label or just random sampling per view.
    # Strategy:
    # 1. Stratify by View Label first (Ensure equal representation of cameras)
    # 2. Then try to pick blocks that improve Label balance (greedy selection)
    
    selected_block_ids = []
    unique_views = df_blocked['view_label'].unique()
    blocks_per_view = target_blocks // len(unique_views)
    
    print(f"Allocating approx {blocks_per_view} blocks per view ({len(unique_views)} views)...")
    
    for view in unique_views:
        view_blocks = df_blocked[df_blocked['view_label'] == view]
        available_ids = view_blocks['temp_block_id'].unique()
        
        # Simple sampling: equidistant sampling throughout the timeline to get diverse times of day
        # e.g. if we have 100 blocks and need 10, take indices 0, 10, 20...
        if len(available_ids) > blocks_per_view:
            idx = np.linspace(0, len(available_ids)-1, blocks_per_view).astype(int)
            picked = available_ids[idx]
        else:
            picked = available_ids # Take all if not enough
            
        selected_block_ids.extend(picked)
        
    # Filter
    subset_df = df_blocked[df_blocked['temp_block_id'].isin(selected_block_ids)].copy()
    
    # Cleanup
    if 'dt' in subset_df.columns:
        subset_df.drop(columns=['dt'], inplace=True)
    subset_df.drop(columns=['temp_block_id'], inplace=True)
    
    print(f"Subset Created: {len(subset_df)} samples")
    print("\nClass Distribution:")
    print(subset_df['congestion_enter_rating'].value_counts())
    print("\nView Distribution:")
    print(subset_df['view_label'].value_counts())
    
    subset_df.to_csv(output_csv, index=False)
    print(f"\nSaved to: {output_csv}")

if __name__ == "__main__":
    base_dir = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev/demos"
    input_path = os.path.join(base_dir, "Train.csv")
    output_path = os.path.join(base_dir, "Train_Balanced_3k.csv")
    
    create_balanced_subset(input_path, output_path)
