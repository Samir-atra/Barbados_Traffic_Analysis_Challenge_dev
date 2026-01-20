"""Block Analysis Script for Traffic Dataset.

This script analyzes the sequential blocks in the traffic dataset to understand
the data distribution and characteristics. It creates a CSV file with details
about each block including:
- View label
- Block ID
- Block length (number of time segments)
- Start and end time segment IDs
- Class distribution within the block
- Time span of the block

This information helps understand the difficulty of the prediction task and
the nature of sequential patterns in the data.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime


def identify_blocks(group):
    """Identifies continuous sequential blocks within a data group.
    
    A block break occurs when time_segment_id is not consecutive.
    
    Args:
        group: A pandas DataFrame group sorted by time_segment_id.
        
    Returns:
        The group with an added 'block_id' column.
    """
    group = group.sort_values('time_segment_id')
    ids = group['time_segment_id'].values
    is_break = np.zeros(len(ids), dtype=int)
    is_break[1:] = (ids[1:] != ids[:-1] + 1).astype(int)
    group['block_id'] = np.cumsum(is_break)
    return group


def analyze_blocks(csv_path, output_path):
    """Analyzes all blocks in the dataset and creates a summary CSV.
    
    Args:
        csv_path: Path to the input CSV file (Train.csv).
        output_path: Path where the analysis CSV will be saved.
    """
    df = pd.read_csv(csv_path)
    
    # Convert video_time to datetime for time analysis
    df['video_time'] = pd.to_datetime(df['video_time'])
    df['date'] = pd.to_datetime(df['date'])
    
    # Congestion mapping
    congestion_map = {
        'free flowing': 0,
        'light delay': 1,
        'moderate delay': 2,
        'heavy delay': 3
    }
    df['enter_id'] = df['congestion_enter_rating'].map(congestion_map).fillna(0).astype(int)
    df['exit_id'] = df['congestion_exit_rating'].map(congestion_map).fillna(0).astype(int)
    
    print(f"Analyzing {len(df)} rows from {csv_path}...")
    
    block_data = []
    
    for view_label, view_group in df.groupby('view_label'):
        view_group = identify_blocks(view_group)
        
        for block_id, block in view_group.groupby('block_id'):
            block_length = len(block)
            
            # Time segment info
            start_segment = block['time_segment_id'].min()
            end_segment = block['time_segment_id'].max()
            
            # Time info
            start_time = block['video_time'].min()
            end_time = block['video_time'].max()
            date = block['date'].iloc[0]
            
            # Calculate time span in minutes
            time_span_minutes = (end_time - start_time).total_seconds() / 60
            
            # Class distribution for enter congestion
            enter_counts = block['enter_id'].value_counts().to_dict()
            enter_free = enter_counts.get(0, 0)
            enter_light = enter_counts.get(1, 0)
            enter_moderate = enter_counts.get(2, 0)
            enter_heavy = enter_counts.get(3, 0)
            
            # Class distribution for exit congestion
            exit_counts = block['exit_id'].value_counts().to_dict()
            exit_free = exit_counts.get(0, 0)
            exit_light = exit_counts.get(1, 0)
            exit_moderate = exit_counts.get(2, 0)
            exit_heavy = exit_counts.get(3, 0)
            
            # Dominant class
            enter_dominant = block['enter_id'].mode().iloc[0] if len(block) > 0 else 0
            exit_dominant = block['exit_id'].mode().iloc[0] if len(block) > 0 else 0
            
            # Class transitions (how many times the class changes)
            enter_transitions = (block['enter_id'].diff().fillna(0) != 0).sum()
            exit_transitions = (block['exit_id'].diff().fillna(0) != 0).sum()
            
            # Entropy of class distribution (measure of class balance)
            enter_probs = np.array([enter_free, enter_light, enter_moderate, enter_heavy]) / block_length
            enter_probs = enter_probs[enter_probs > 0]  # Remove zeros for log
            enter_entropy = -np.sum(enter_probs * np.log2(enter_probs)) if len(enter_probs) > 0 else 0
            
            # Is the block usable for training? (needs at least seq_len + 1 = 16 samples)
            is_usable = block_length >= 16
            
            # Can create sequences (number of possible windows)
            n_windows = max(0, block_length - 15)
            
            block_data.append({
                'view_label': view_label,
                'block_id': block_id,
                'length': block_length,
                'n_possible_windows': n_windows,
                'is_usable': is_usable,
                'start_segment_id': start_segment,
                'end_segment_id': end_segment,
                'date': date.strftime('%Y-%m-%d'),
                'start_time': start_time.strftime('%H:%M:%S'),
                'end_time': end_time.strftime('%H:%M:%S'),
                'time_span_minutes': round(time_span_minutes, 2),
                'enter_free_flowing': enter_free,
                'enter_light_delay': enter_light,
                'enter_moderate_delay': enter_moderate,
                'enter_heavy_delay': enter_heavy,
                'exit_free_flowing': exit_free,
                'exit_light_delay': exit_light,
                'exit_moderate_delay': exit_moderate,
                'exit_heavy_delay': exit_heavy,
                'enter_dominant_class': enter_dominant,
                'exit_dominant_class': exit_dominant,
                'enter_transitions': enter_transitions,
                'exit_transitions': exit_transitions,
                'enter_entropy': round(enter_entropy, 4),
            })
    
    # Create DataFrame
    block_df = pd.DataFrame(block_data)
    
    # Sort by view_label and block_id
    block_df = block_df.sort_values(['view_label', 'block_id']).reset_index(drop=True)
    
    # Save to CSV
    block_df.to_csv(output_path, index=False)
    print(f"Saved block analysis to: {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("BLOCK ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal blocks: {len(block_df)}")
    print(f"Usable blocks (length >= 16): {block_df['is_usable'].sum()}")
    print(f"Total possible training windows: {block_df['n_possible_windows'].sum()}")
    
    print("\nBlocks per view:")
    print(block_df.groupby('view_label')['block_id'].count().to_string())
    
    print("\nBlock length statistics:")
    print(f"  Min:    {block_df['length'].min()}")
    print(f"  Max:    {block_df['length'].max()}")
    print(f"  Mean:   {block_df['length'].mean():.2f}")
    print(f"  Median: {block_df['length'].median():.2f}")
    print(f"  Std:    {block_df['length'].std():.2f}")
    
    print("\nBlock length distribution:")
    length_bins = [0, 10, 16, 30, 50, 100, 200, 500, 1000, float('inf')]
    length_labels = ['1-10', '11-15', '16-30', '31-50', '51-100', 
                     '101-200', '201-500', '501-1000', '1000+']
    block_df['length_bin'] = pd.cut(block_df['length'], bins=length_bins, 
                                     labels=length_labels, right=False)
    print(block_df['length_bin'].value_counts().sort_index().to_string())
    
    print("\nClass distribution (enter congestion) across all blocks:")
    total_enter = (block_df['enter_free_flowing'].sum() + 
                   block_df['enter_light_delay'].sum() +
                   block_df['enter_moderate_delay'].sum() + 
                   block_df['enter_heavy_delay'].sum())
    print(f"  Free Flowing:   {block_df['enter_free_flowing'].sum():5d} ({100*block_df['enter_free_flowing'].sum()/total_enter:.1f}%)")
    print(f"  Light Delay:    {block_df['enter_light_delay'].sum():5d} ({100*block_df['enter_light_delay'].sum()/total_enter:.1f}%)")
    print(f"  Moderate Delay: {block_df['enter_moderate_delay'].sum():5d} ({100*block_df['enter_moderate_delay'].sum()/total_enter:.1f}%)")
    print(f"  Heavy Delay:    {block_df['enter_heavy_delay'].sum():5d} ({100*block_df['enter_heavy_delay'].sum()/total_enter:.1f}%)")
    
    print("\nDominant class distribution (enter):")
    dom_map = {0: 'Free Flowing', 1: 'Light Delay', 2: 'Moderate Delay', 3: 'Heavy Delay'}
    print(block_df['enter_dominant_class'].map(dom_map).value_counts().to_string())
    
    print("\nClass transition statistics (enter):")
    print(f"  Min transitions:  {block_df['enter_transitions'].min()}")
    print(f"  Max transitions:  {block_df['enter_transitions'].max()}")
    print(f"  Mean transitions: {block_df['enter_transitions'].mean():.2f}")
    
    print("\nEntropy statistics (measure of class balance within blocks):")
    print(f"  Min entropy:  {block_df['enter_entropy'].min():.4f}")
    print(f"  Max entropy:  {block_df['enter_entropy'].max():.4f}")
    print(f"  Mean entropy: {block_df['enter_entropy'].mean():.4f}")
    print("  (Higher entropy = more balanced class distribution)")
    
    return block_df


def main():
    """Main execution function."""
    base = "/teamspace/studios/this_studio/Barbados_Traffic_Analysis_Challenge_dev"
    if not os.path.exists(base):
        base = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev"
    
    train_path = os.path.join(base, "demos/Train.csv")
    output_path = os.path.join(base, "analytics/block_analysis.csv")
    
    # Ensure analytics directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if os.path.exists(train_path):
        analyze_blocks(train_path, output_path)
    else:
        print(f"Error: Could not find training data at {train_path}")


if __name__ == "__main__":
    main()
