
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

# Add data_processing to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data_processing'))
from chunked_data_loader import create_raw_sequences_chunked, SEQ_LEN
from esn_traffic import ESNClassifier

def sweep_esn():
    base = "/home/samer/Desktop/competitions/Barbados_Traffic_Analysis_Challenge_dev"
    train_path = os.path.join(base, "demos/Train.csv")
    
    print("Loading data for sweep...")
    X_train, y_train, X_val, y_val = create_raw_sequences_chunked(
        train_path, val_split=0.2, seq_len=SEQ_LEN)
    
    # Define Sweep Grid
    radii = [0.95, 1.1, 1.25]
    leaks = [0.1, 0.2, 0.4]
    alphas = [0.1, 1.0, 10.0]
    
    results = []
    
    input_dim = X_train.shape[2]
    
    # Balance weights
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))

    for rho in radii:
        for lr in leaks:
            for a in alphas:
                print(f"\n--- Testing: Radius={rho}, Leak={lr}, Alpha={a} ---")
                esn = ESNClassifier(
                    input_dim=input_dim,
                    reservoir_dim=3000,
                    spectral_radius=rho,
                    leak_rate=lr,
                    alpha=a,
                    state_avg_steps=3,
                    seed=42
                )
                
                esn.fit(X_train, y_train, class_weights=class_weight_dict)
                
                v_pred = esn.predict(X_val)
                v_acc = accuracy_score(y_val, v_pred)
                v_f1 = f1_score(y_val, v_pred, average='macro')
                
                t_pred = esn.predict(X_train)
                t_acc = accuracy_score(y_train, t_pred)
                
                print(f"Result: Val Acc={v_acc:.4f}, Val F1={v_f1:.4f}, Train Acc={t_acc:.4f}")
                
                results.append({
                    'radius': rho, 'leak': lr, 'alpha': a,
                    'val_acc': v_acc, 'val_f1': v_f1, 'train_acc': t_acc
                })

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values('val_f1', ascending=False)
    print("\n" + "="*50)
    print("TOP 5 CONFIGURATIONS")
    print("="*50)
    print(df_res.head(5))

    best = df_res.iloc[0]
    return best

if __name__ == "__main__":
    sweep_esn()
