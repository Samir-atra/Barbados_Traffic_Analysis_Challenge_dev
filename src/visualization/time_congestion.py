"""Module to visualize traffic congestion over time.

This script extracts data from the Train.csv file and creates a visualization
to analyze the relationship between the time of day and the entrance congestion
level.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_time_congestion_plot(csv_path: str, output_path: str) -> None:
    """Generates a plot of entrance congestion rate against video time.

    Args:
        csv_path: Absolute path to the training CSV file.
        output_path: Absolute path where the resulting plot will be saved.
    """
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Convert video_time to datetime objects
    df['video_time'] = pd.to_datetime(df['video_time'])

    # Map categorical congestion ratings to numerical levels for visualization
    # 1: free flowing, 2: light delay, 3: moderate delay, 4: heavy delay
    congestion_map = {
        'free flowing': 1,
        'light delay': 2,
        'moderate delay': 3,
        'heavy delay': 4
    }
    df['congestion_level'] = df['congestion_exit_rating'].map(congestion_map)

    # Extract the decimal time (hours + minutes/60) for a continuous x-axis
    df['time_decimal'] = df['video_time'].dt.hour + df['video_time'].dt.minute / 60.0

    # Set up the plotting style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 7))

    # Plot the raw data points with transparency to show density
    sns.scatterplot(
        data=df,
        x='time_decimal',
        y='congestion_level',
        alpha=0.1,
        s=10,
        color='blue',
        label='Data points'
    )

    # Calculate and plot the average congestion level across time intervals
    # We use a rolling window or a regression line to show the trend
    sns.lineplot(
        data=df,
        x='time_decimal',
        y='congestion_level',
        color='red',
        linewidth=2,
        label='Mean Trend'
    )

    # Customize the plot
    plt.title('Entrance Congestion Level vs. Time of Day', fontsize=18)
    plt.xlabel('Time of Day (Hour)', fontsize=14)
    plt.ylabel('Congestion Level', fontsize=14)

    # Set y-axis ticks to match the categories
    plt.yticks(
        ticks=[1, 2, 3, 4],
        labels=['Free Flowing', 'Light Delay', 'Moderate Delay', 'Heavy Delay']
    )

    # Set x-axis ticks to show hours (0-23)
    plt.xticks(range(25))
    plt.xlim(0, 24)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Successfully generated visualization and saved to: {output_path}")


def main():
    """Main entry point for the visualization script."""
    # Use absolute paths as required
    base_dir = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev"
    csv_file = os.path.join(base_dir, "demos/Train.csv")
    
    # Using 'analytics' directory as per user rules
    output_file = os.path.join(base_dir, "analytics/time_exit_congestion_plot.png")

    if os.path.exists(csv_file):
        generate_time_congestion_plot(csv_file, output_file)
    else:
        print(f"Error: Could not find training data at {csv_file}")


if __name__ == "__main__":
    main()
