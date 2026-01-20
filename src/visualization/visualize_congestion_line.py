
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_full_range_congestion(data_path, output_dir):
    """Generates continuous line plots for the full range of data.

    Args:
        data_path: Path to the Train.csv file.
        output_dir: Directory to save the generated plots.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    df = pd.read_csv(data_path)
    df['datetimestamp_start'] = pd.to_datetime(df['datetimestamp_start'])
    df = df.sort_values('datetimestamp_start')

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
    
    # We will generate one plot per view_label to show the full range clearly
    view_labels = df['view_label'].unique()
    
    print(f"Generating full range plots for {len(view_labels)} locations...")

    for view in view_labels:
        view_df = df[df['view_label'] == view].copy()
        
        plt.figure(figsize=(20, 8))
        
        # Plot with connected dots
        plt.plot(view_df['datetimestamp_start'], view_df['congestion_level'], 
                 marker='o', linestyle='-', markersize=4, linewidth=1, alpha=0.7, label=view)
        
        plt.title(f'Continuous Congestion Trend - {view} (Full Range)', fontsize=16)
        plt.xlabel('Date and Time', fontsize=12)
        plt.ylabel('Congestion Level', fontsize=12)
        plt.ylim(-0.5, 3.5)
        plt.yticks([0, 1, 2, 3], ['Free Flowing', 'Light Delay', 'Moderate Delay', 'Heavy Delay'])
        
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot
        safe_view = view.replace('#', 'No').replace(' ', '_')
        plot_path = os.path.join(output_dir, f'full_range_{safe_view}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved full range plot for {view} to {plot_path}")

    # Also generate a summary plot with all views
    plt.figure(figsize=(24, 10))
    for view in view_labels:
        view_df = df[df['view_label'] == view].copy()
        plt.plot(view_df['datetimestamp_start'], view_df['congestion_level'], 
                 marker='.', linestyle='-', markersize=2, linewidth=0.8, alpha=0.6, label=view)
    
    plt.title('Multi-Location Congestion Trend (Full Range)', fontsize=18)
    plt.xlabel('Date and Time', fontsize=14)
    plt.ylabel('Congestion Level', fontsize=14)
    plt.ylim(-0.5, 3.5)
    plt.yticks([0, 1, 2, 3], ['Free Flowing', 'Light Delay', 'Moderate Delay', 'Heavy Delay'])
    plt.legend(title='Location', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    summary_path = os.path.join(output_dir, 'full_range_combined_trend.png')
    plt.savefig(summary_path)
    plt.close()
    print(f"Saved combined trend plot to {summary_path}")

if __name__ == "__main__":
    TRAIN_CSV = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev/demos/Train.csv"
    # Using analytics_data directory as per rule 4
    ANALYTICS_DATA_DIR = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev/analytics_data/congestion_line_plots"
    
    visualize_full_range_congestion(TRAIN_CSV, ANALYTICS_DATA_DIR)
