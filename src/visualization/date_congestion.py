"""Module to visualize traffic congestion over different dates.

This script extracts data from the Train.csv file and creates a visualization
to analyze how the entrance congestion level varies across different dates
in the dataset.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_date_congestion_plot(csv_path: str, output_path: str) -> None:
    """Generates a plot of entrance congestion rate against date.

    Args:
        csv_path: Absolute path to the training CSV file.
        output_path: Absolute path where the resulting plot will be saved.
    """
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Convert date to datetime objects for proper sorting and plotting
    df['date'] = pd.to_datetime(df['date'])

    # Map categorical congestion ratings to numerical levels for visualization
    # 1: free flowing, 2: light delay, 3: moderate delay, 4: heavy delay
    congestion_map = {
        'free flowing': 1,
        'light delay': 2,
        'moderate delay': 3,
        'heavy delay': 4
    }
    df['congestion_level'] = df['congestion_exit_rating'].map(congestion_map)

    # Calculate the average daily congestion level
    daily_avg = df.groupby('date')['congestion_level'].mean().reset_index()

    # Set up the plotting style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 8))

    # Plot daily distribution using a boxplot to see the spread on each date
    sns.boxplot(
        data=df,
        x='date',
        y='congestion_level',
        color='skyblue',
        showfliers=False
    )

    # Overlaid line plot for the mean trend across dates
    sns.lineplot(
        data=daily_avg,
        x=daily_avg.index,
        y='congestion_level',
        marker='o',
        color='red',
        linewidth=2,
        label='Daily Mean'
    )

    # Customize the plot
    plt.title('Daily Exit Congestion Levels', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Congestion Level', fontsize=14)

    # Set y-axis ticks to match the categories
    plt.yticks(
        ticks=[1, 2, 3, 4],
        labels=['Free Flowing', 'Light Delay', 'Moderate Delay', 'Heavy Delay']
    )

    # Formatting x-axis dates
    dates = sorted(df['date'].unique())
    plt.xticks(
        ticks=range(len(dates)),
        labels=[d.strftime('%Y-%m-%d') for d in dates],
        rotation=45
    )

    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Successfully generated date visualization and saved to: {output_path}")


def main():
    """Main entry point for the date-congestion visualization script."""
    # Use absolute paths as required
    base_dir = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev"
    csv_file = os.path.join(base_dir, "demos/Train.csv")
    
    # Using 'analytics' directory as per user rules
    output_file = os.path.join(base_dir, "analytics/date_exit_congestion_plot.png")

    if os.path.exists(csv_file):
        generate_date_congestion_plot(csv_file, output_file)
    else:
        print(f"Error: Could not find training data at {csv_file}")


if __name__ == "__main__":
    main()
