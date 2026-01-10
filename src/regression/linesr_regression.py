"""Module to perform linear regression on traffic congestion over time.

This script performs linear regression to model the relationship between the 
time of day and the entrance congestion level using the linear equation y = mx + b.
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns


def perform_linear_regression(csv_path: str, output_plot_path: str) -> None:
    """Performs linear regression and visualizes the result.

    Args:
        csv_path: Absolute path to the training CSV file.
        output_plot_path: Absolute path where the resulting plot will be saved.
    """
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Convert video_time to datetime and then to decimal hours
    df['video_time'] = pd.to_datetime(df['video_time'])
    df['time_decimal'] = df['video_time'].dt.hour + df['video_time'].dt.minute / 60.0 + df['video_time'].dt.second / 3600.0

    # Map categorical congestion ratings to numerical levels
    congestion_map = {
        'free flowing': 1,
        'light delay': 2,
        'moderate delay': 3,
        'heavy delay': 4
    }
    df['congestion_level'] = df['congestion_exit_rating'].map(congestion_map)

    # Drop any rows with NaN values in relevant columns
    df = df.dropna(subset=['time_decimal', 'congestion_level'])

    # Prepare features and target
    X = df[['time_decimal']].values
    y = df['congestion_level'].values

    # Initialize and fit the model
    model = LinearRegression()
    model.fit(X, y)

    # Get the coefficients for the line equation y = mx + b
    m = model.coef_[0]
    b = model.intercept_

    print("\n--- Linear Regression Results ---")
    print(f"Line Equation: y = {m:.4f}x + {b:.4f}")
    print(f"Slope (m): {m:.4f}")
    print(f"Intercept (b): {b:.4f}")
    print("---------------------------------\n")

    # Generate predictions for the trend line
    x_range = np.linspace(0, 24, 100).reshape(-1, 1)
    y_pred = model.predict(x_range)

    # Set up the plotting style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # Scatter plot of actual data points
    plt.scatter(df['time_decimal'], df['congestion_level'], alpha=0.05, s=5, color='blue', label='Actual Data')

    # Regression line
    plt.plot(x_range, y_pred, color='red', linewidth=3, label=f'Linear Regression: y = {m:.2f}x + {b:.2f}')

    # Customize the plot
    plt.title('Linear Regression: Exit Congestion vs. Time of Day', fontsize=16)
    plt.xlabel('Time of Day (Hour)', fontsize=12)
    plt.ylabel('Congestion Level', fontsize=12)
    plt.yticks([1, 2, 3, 4], ['Free Flowing', 'Light Delay', 'Moderate Delay', 'Heavy Delay'])
    plt.xticks(range(25))
    plt.xlim(0, 24)
    plt.legend()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    
    print(f"Regression plot saved to: {output_plot_path}")


def main():
    """Main execution function."""
    base_dir = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev"
    csv_file = os.path.join(base_dir, "demos/Train.csv")
    
    # Save the visualization to the analytics directory
    output_file = os.path.join(base_dir, "analytics/exit_congestion_regression.png")

    if os.path.exists(csv_file):
        perform_linear_regression(csv_file, output_file)
    else:
        print(f"Error: Could not find training data at {csv_file}")


if __name__ == "__main__":
    main()
