"""Module to visualize traffic congestion enter rating over time for each day.

This script loads the training data, processes the timestamps, and generates
plots for each day showing the congestion level at different locations.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_congestion(data_path, output_dir):
    """Generates congestion plots for each day in the dataset.

    Args:
        data_path: Path to the Train.csv file.
        output_dir: Directory to save the generated plots.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    df = pd.read_csv(data_path)

    # Convert timestamps to datetime objects
    df['datetimestamp_start'] = pd.to_datetime(df['datetimestamp_start'])
    df['date'] = df['datetimestamp_start'].dt.date
    df['time'] = df['datetimestamp_start'].dt.time
    # Create a decimal hour for plotting on the x-axis
    df['hour_float'] = df['datetimestamp_start'].dt.hour + df['datetimestamp_start'].dt.minute / 60.0

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
    plt.rcParams['figure.figsize'] = (15, 8)

    # Get unique days
    unique_days = sorted(df['date'].unique())

    print(f"Generating plots for {len(unique_days)} days...")

    for day in unique_days:
        day_df = df[df['date'] == day].copy()
        day_df = day_df.sort_values('datetimestamp_start')

        plt.figure()
        # Plot each view_label (location) separately
        sns.lineplot(
            data=day_df, 
            x='hour_float', 
            y='congestion_level', 
            hue='view_label',
            marker='o',
            alpha=0.7
        )

        plt.title(f'Congestion Enter Rating - {day}', fontsize=16)
        plt.xlabel('Hour of the Day', fontsize=12)
        plt.ylabel('Congestion Level (0: Free, 3: Heavy)', fontsize=12)
        plt.ylim(-0.5, 3.5)
        plt.yticks([0, 1, 2, 3], ['Free Flowing', 'Light Delay', 'Moderate Delay', 'Heavy Delay'])
        plt.xticks(range(0, 25))
        plt.legend(title='Location', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Save the plot
        safe_date = str(day).replace('-', '_')
        plot_path = os.path.join(output_dir, f'congestion_{safe_date}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot for {day} to {plot_path}")

if __name__ == "__main__":
    TRAIN_CSV = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev/kaggle_data/Train.csv"
    ANALYTICS_DIR = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev/analytics/congestion_plots"
    
    visualize_congestion(TRAIN_CSV, ANALYTICS_DIR)
