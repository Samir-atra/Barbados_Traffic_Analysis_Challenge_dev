import os
import sys
import numpy as np
import pandas as pd

# Add the source directory to path
sys.path.append("/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev/src/forward-forward")
from ff_restructured import create_sequential_dataset

def main():
    base = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev"
    train_path = os.path.join(base, "demos/Train.csv")
    
    print("Loading data and generating features...")
    # Following the most recent logic in ff_restructured.py
    X, y = create_sequential_dataset(train_path)
    
    print(f"\nTotal samples: {len(X)}")
    print(f"Feature vector shape: {X.shape[1]}")
    
    print("\nFirst 3 sample feature vectors:")
    np.set_printoptions(precision=4, suppress=True)
    for i in range(min(len(X), 3)):
        print(f"\nSample {i}:")
        print(f"Full Vector: {X[i]}")
        print(f"  Hour (norm):    {X[i, 16]:.4f}")
        print(f"  Minute (norm):  {X[i, 17]:.4f}")
        print(f"  DayOfWeek(norm):{X[i, 18]:.4f}")
        print(f"  Segment ID:     {X[i, 19]:.4f}")
        print(f"  View ID:        {X[i, 20]:.0f}")
        print(f"  View One-hot:   {X[i, 21:25]}")

if __name__ == "__main__":
    main()
