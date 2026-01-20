
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_downsampled_congestion(data_path, output_dir, window_size=10):
    """Generates downsampled line plots (average of N elements).

    Args:
        data_path: Path to the Train.csv file.
        output_dir: Directory to save the generated plots.
        window_size: Number of sequential elements to average.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    df = pd.read_csv(data_path)
    df['datetimestamp_start'] = pd.to_datetime(df['datetimestamp_start'])
    
    # Define the mapping for congestion levels
    rating_map = {
        'free flowing': 0,
        'light delay': 1,
        'moderate delay': 2,
        'heavy delay': 3
    }
    df['congestion_level'] = df['congestion_enter_rating'].map(rating_map)

    # Set the style
    sns.set_theme(style="whitegrid")
    
    view_labels = df['view_label'].unique()
    
    print(f"Generating downsampled plots (window={window_size}) for {len(view_labels)} locations...")

    for view in view_labels:
        # Get data for this view and ensure it is sorted by time
        view_df = df[df['view_label'] == view].sort_values('datetimestamp_start').copy()
        
        # Create a grouping key for every 10 elements
        # This ensures we take blocks of 10 sequential elements
        view_df['group'] = np.arange(len(view_df)) // window_size
        
        # Calculate the average of each group
        # For the timestamp, we take the mean (middle point of the window)
        downsampled = view_df.groupby('group').agg({
            'datetimestamp_start': 'mean',
            'congestion_level': 'mean'
        }).reset_index()
        
        plt.figure(figsize=(20, 8))
        
        # Plot the averages with connected dots
        plt.plot(downsampled['datetimestamp_start'], downsampled['congestion_level'], 
                 marker='o', linestyle='-', markersize=6, linewidth=1.5, alpha=0.8, color='teal', label=f"{view} (Avg of {window_size})")
        
        plt.title(f'Downsampled Congestion Trend - {view} (Window={window_size})', fontsize=16)
        plt.xlabel('Date and Time', fontsize=12)
        plt.ylabel('Average Congestion Level', fontsize=12)
        plt.ylim(-0.1, 3.1)
        
        # Add horizontal lines for reference
        for y, label in zip([0, 1, 2, 3], ['Free', 'Light', 'Mod', 'Heavy']):
            plt.axhline(y=y, color='gray', linestyle='--', alpha=0.3)
            
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot
        safe_view = view.replace('#', 'No').replace(' ', '_')
        plot_path = os.path.join(output_dir, f'downsampled_{window_size}_{safe_view}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved downsampled plot for {view} to {plot_path}")

    # Combined summary plot
    plt.figure(figsize=(24, 10))
    for view in view_labels:
        view_df = df[df['view_label'] == view].sort_values('datetimestamp_start').copy()
        view_df['group'] = np.arange(len(view_df)) // window_size
        downsampled = view_df.groupby('group').agg({
            'datetimestamp_start': 'mean',
            'congestion_level': 'mean'
        }).reset_index()
        
        plt.plot(downsampled['datetimestamp_start'], downsampled['congestion_level'], 
                 marker='.', linestyle='-', markersize=4, linewidth=1, alpha=0.7, label=view)
    
    plt.title(f'Multi-Location Downsampled Congestion (Avg of {window_size})', fontsize=18)
    plt.xlabel('Date and Time', fontsize=14)
    plt.ylabel('Average Congestion Level', fontsize=14)
    plt.ylim(-0.1, 3.1)
    plt.legend(title='Location', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    summary_path = os.path.join(output_dir, f'downsampled_{window_size}_combined.png')
    plt.savefig(summary_path)
    plt.close()
    print(f"Saved combined downsampled trend to {summary_path}")

if __name__ == "__main__":
    TRAIN_CSV = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev/demos/Train.csv"
    ANALYTICS_DATA_DIR = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev/analytics_data/congestion_downsampled"
    
    visualize_downsampled_congestion(TRAIN_CSV, ANALYTICS_DATA_DIR, window_size=20)
