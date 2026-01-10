"""Module to count the occurrences of congestion rating classes in Train.csv."""

import pandas as pd
import os

def count_classes(csv_path):
    """
    Counts the occurrences of congestion rating classes in the provided CSV file.
    
    Args:
        csv_path (str): The absolute path to the Train.csv file.
    """
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Columns of interest
    cols = ['congestion_enter_rating', 'congestion_exit_rating']
    
    print(f"--- Class Distribution in {os.path.basename(csv_path)} ---\n")
    
    for col in cols:
        if col in df.columns:
            print(f"Distribution for {col}:")
            counts = df[col].value_counts()
            percentages = df[col].value_counts(normalize=True) * 100
            
            # Combine counts and percentages for a cleaner view
            distribution = pd.concat([counts, percentages], axis=1)
            distribution.columns = ['Count', 'Percentage (%)']
            print(distribution)
            print("-" * 30)
        else:
            print(f"Warning: Column {col} not found in the dataset.")

if __name__ == "__main__":
    base_dir = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev"
    train_csv = os.path.join(base_dir, "demos/Train.csv")
    count_classes(train_csv)
