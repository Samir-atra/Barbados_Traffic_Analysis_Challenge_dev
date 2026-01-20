
import os
import sys
import numpy as np
import pandas as pd

# Add src to path to import data loader
base_dir = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev"
sys.path.insert(0, os.path.join(base_dir, 'src/data_processing'))

from chunked_data_loader import create_raw_sequences_chunked

def export_samples():
    """Exports samples of chunked features and labels to CSV."""
    train_path = os.path.join(base_dir, "demos/Train.csv")
    
    print(f"Loading data from {train_path}...")
    X_train, y_train, _, _ = create_raw_sequences_chunked(train_path)
    
    # Feature names based on get_features_and_labels implementation (12 features)
    feature_names = [
        'hour_norm', 'minute_norm', 'seg_id_norm',
        'view_id_norm',
        'view_0', 'view_1', 'view_2', 'view_3',
        'sig_0', 'sig_1', 'sig_2', 'sig_3'
    ]
    
    samples = []
    classes = np.unique(y_train)
    
    for cls in classes:
        print(f"Extracting 5 samples for class {cls}...")
        idx = np.where(y_train == cls)[0]
        selected_idx = idx[:5]
        
        for i in selected_idx:
            # Each X_train[i] is (seq_len, n_features)
            # We take the LAST timestep in the sequence as the representative feature set for that label
            feat_vec = X_train[i][-1] 
            
            sample_dict = {f"feat_{name}": val for name, val in zip(feature_names, feat_vec)}
            sample_dict['label'] = cls
            sample_dict['sample_index'] = i
            samples.append(sample_dict)
            
    df_samples = pd.DataFrame(samples)
    
    output_dir = os.path.join(base_dir, "analytics_data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "model_input_samples.csv")
    
    df_samples.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df_samples)} samples to {output_path}")

if __name__ == "__main__":
    export_samples()
