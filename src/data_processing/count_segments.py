"""Script to count continuous sequential segments in the Barbados Traffic dataset.

This script analyzes the 'time_segment_id' (extracted from ID_enter and ID_exit)
to identify and count how many continuous blocks of sequential data (N, N+1) 
exist for each location (view_label), and provides the size of each segment.
"""

import pandas as pd
import os
import numpy as np


def count_sequential_blocks(csv_path: str) -> None:
    """Counts sequential segments and their sizes for each view_label.

    Args:
        csv_path: Absolute path to the Train.csv file.
    """
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    # Load the dataset
    df = pd.read_csv(csv_path)

    # List to store summary for each view
    summary_results = []
    # List to store individual segment details
    detailed_segments = []

    # Process each location (view_label) independently
    for label, group in df.groupby('view_label'):
        # Sort by time_segment_id to check sequentiality
        ids = group['time_segment_id'].sort_values().values
        
        if len(ids) == 0:
            summary_results.append({
                'View Label': label,
                'Total Samples': 0,
                'Sequential Blocks': 0,
                'Avg Segment Size': 0
            })
            continue

        # Identify changes in sequentiality
        # 0 if sequential (id[i] == id[i-1] + 1), otherwise 1
        is_break = np.zeros(len(ids), dtype=int)
        is_break[1:] = (ids[1:] != ids[:-1] + 1).astype(int)
        
        # Cumulative sum creates unique IDs for each sequential block
        block_ids = np.cumsum(is_break)
        
        # Calculate lengths of each block
        segment_lengths = pd.Series(block_ids).value_counts().sort_index().values
        
        # Store detailed segment information
        for i, length in enumerate(segment_lengths):
            detailed_segments.append({
                'View Label': label,
                'Segment Index': i,
                'Length': length
            })

        summary_results.append({
            'View Label': label,
            'Total Samples': len(ids),
            'Sequential Blocks': len(segment_lengths),
            'Min Segment Size': segment_lengths.min(),
            'Max Segment Size': segment_lengths.max(),
            'Avg Segment Size': round(segment_lengths.mean(), 2),
            'Segment Lengths': segment_lengths.tolist()
        })

    # Create summary and detailed DataFrames
    summary_df = pd.DataFrame(summary_results)
    detailed_df = pd.DataFrame(detailed_segments)

    print("\n" + "="*80)
    print("SEQUENTIAL SEGMENT ANALYSIS SUMMARY")
    print("="*80)
    # Printing without the full list of Segment Lengths for readability
    print(summary_df.drop(columns=['Segment Lengths']).to_string(index=False))
    print("="*80)
    
    # Detailed display of some segments for context
    print("\n" + "="*80)
    print("SAMPLES OF INDIVIDUAL SEGMENT LENGTHS (First 10 for Norman Niles #1)")
    print("="*80)
    print(detailed_df[detailed_df['View Label'] == "Norman Niles #1"].head(10).to_string(index=False))
    print("="*80)
    
    # Save results to the analytics directory
    analytics_dir = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev/analytics"
    os.makedirs(analytics_dir, exist_ok=True)
    
    summary_csv = os.path.join(analytics_dir, "segment_summary.csv")
    detailed_csv = os.path.join(analytics_dir, "individual_segment_counts.csv")
    
    summary_df.to_csv(summary_csv, index=False)
    detailed_df.to_csv(detailed_csv, index=False)
    
    print(f"\nSummary saved to: {summary_csv}")
    print(f"Detailed counts per segment saved to: {detailed_csv}\n")


def main():
    """Main execution entry point."""
    base_dir = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev"
    train_csv = os.path.join(base_dir, "demos/Train.csv")
    
    count_sequential_blocks(train_csv)


if __name__ == "__main__":
    main()
