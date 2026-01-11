import pandas as pd
import numpy as np
import os

def count_sequences(csv_path):
    if not os.path.exists(csv_path):
        return f"File not found: {csv_path}"
    
    df = pd.read_csv(csv_path)
    block_lengths = []
    
    for label, group in df.groupby('view_label'):
        group = group.sort_values('time_segment_id')
        ids = group['time_segment_id'].values
        
        if len(ids) == 0: continue
        
        # Identify breaks in continuity
        is_break = np.zeros(len(ids), dtype=int)
        is_break[1:] = (ids[1:] != ids[:-1] + 1).astype(int)
        block_ids = np.cumsum(is_break)
        
        # Calculate size of each block
        chunk_sizes = pd.Series(block_ids).value_counts().values
        block_lengths.extend(chunk_sizes)
    
    return pd.Series(block_lengths).value_counts().sort_index()

def main():
    base = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev"
    train_path = os.path.join(base, "demos/Train.csv")
    test_path = os.path.join(base, "demos/TestInputSegments.csv")
    
    train_counts = count_sequences(train_path)
    test_counts = count_sequences(test_path)
    
    print("\n### Sequence Length Analysis")
    
    # Combine into a single table
    all_lengths = sorted(set(train_counts.index) | set(test_counts.index))
    
    print(f"{'Length':<10} | {'Train Count':<15} | {'Test Count':<15}")
    print("-" * 45)
    for length in all_lengths:
        tr = train_counts.get(length, 0)
        te = test_counts.get(length, 0)
        print(f"{length:<10} | {tr:<15} | {te:<15}")

if __name__ == "__main__":
    main()
