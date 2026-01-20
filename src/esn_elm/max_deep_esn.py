
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin

class DeepESN(BaseEstimator, ClassifierMixin):
    """
    Deep Echo State Network (DeepESN) implementation.
    Consists of a stack of reservoir layers. Each layer feeds into the next.
    The final state used for prediction is the concatenation of states from all layers.
    """
    def __init__(self, input_dim=1280, n_layers=2, res_dim=1000, spectral_radius=0.9, leak_rate=0.2, ridge_alpha=1.0, random_state=42):
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.res_dim = res_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state
        
        # Initialize Architecture
        self.layers = [] # List of dicts {'W_in', 'W_res'}
        rng = np.random.RandomState(self.random_state)
        
        for i in range(n_layers):
            # Input dimension for layer i: 
            # Layer 0 takes original input (input_dim)
            # Layer >0 takes state of previous layer (res_dim)
            curr_input_dim = input_dim if i == 0 else res_dim
            
            # Input weights
            W_in = rng.uniform(-1, 1, (res_dim, curr_input_dim))
            
            # Reservoir weights (Sparse)
            W_res = rng.uniform(-1, 1, (res_dim, res_dim))
            mask = rng.rand(res_dim, res_dim) > 0.95
            W_res[mask] = 0
            
            # Spectral Radius Scaling
            try:
                eigenvalues = np.linalg.eigvals(W_res)
                max_eig = np.max(np.abs(eigenvalues))
                if max_eig > 0:
                    W_res *= (self.spectral_radius / max_eig)
            except:
                W_res *= 0.9 # Fallback
                
            self.layers.append({
                'W_in': W_in,
                'W_res': W_res
            })
            
        self.readout = Ridge(alpha=self.ridge_alpha)
        
    def get_states_sequence(self, input_seq):
        """
        Processes a sequence (T, input_dim) through the Deep ESN stack.
        Returns: (T, n_layers * res_dim) - Concatenated states of all layers
        """
        T = input_seq.shape[0]
        
        # We need to store full sequence of states for each layer to feed to the next
        # layer_states: prediction input for next layer
        prev_layer_seq = input_seq # Start with actual input
        
        all_layers_collected_states = [] # To be concatenated for readout
        
        for i, layer in enumerate(self.layers):
            W_in = layer['W_in']
            W_res = layer['W_res']
            
            current_layer_states = np.zeros((T, self.res_dim))
            x = np.zeros(self.res_dim)
            
            for t in range(T):
                u = prev_layer_seq[t]
                
                # Standard ESN Equestion: x(t) = (1-a)x(t-1) + a*tanh(Win*u + Wres*x(t-1))
                pre = np.dot(W_in, u) + np.dot(W_res, x)
                update = np.tanh(pre)
                x = (1 - self.leak_rate) * x + self.leak_rate * update
                
                current_layer_states[t] = x
            
            all_layers_collected_states.append(current_layer_states)
            prev_layer_seq = current_layer_states # Output of this layer is input to next
            
        # Concatenate all layers: (T, n_layers * res_dim)
        final_states = np.hstack(all_layers_collected_states)
        return final_states

    def fit(self, blocks, compute_metrics=True):
        """
        Trains the execution readout on sequential blocks.
        blocks: List of (X_seq, y_seq, ...)
        """
        all_states = []
        all_targets = []
        
        print(f"DeepESN: Training on {len(blocks)} blocks (Layers={self.n_layers}, ResDim={self.res_dim})...")
        
        for idx, (X_seq, y_seq, _) in enumerate(blocks):
            states_seq = self.get_states_sequence(X_seq)
            all_states.append(states_seq)
            all_targets.append(y_seq)
            
        # Stack
        X_train_res = np.vstack(all_states)
        y_train_flat = np.concatenate(all_targets)
        
        self.readout.fit(X_train_res, y_train_flat)
        
        if compute_metrics:
            y_pred = self.readout.predict(X_train_res)
            y_pred_class = np.round(np.clip(y_pred, 0, 3)).astype(int)
            acc = accuracy_score(y_train_flat, y_pred_class)
            f1 = f1_score(y_train_flat, y_pred_class, average='macro')
            print(f"[TRAIN] DeepESN Accuracy: {acc:.4f} | F1-Macro: {f1:.4f}")
            
    def predict(self, blocks):
        all_preds = []
        for X_seq, _, _ in blocks:
            states_seq = self.get_states_sequence(X_seq)
            preds_seq = self.readout.predict(states_seq)
            all_preds.append(preds_seq)
        return all_preds

# --- Simple Unit Test (Mock Data) ---
if __name__ == "__main__":
    print("Testing DeepESN Implementation...")
    
    # Mock Data: 5 Blocks of length 10, Input dim 128
    dummy_blocks = []
    for _ in range(5):
        X = np.random.rand(10, 128)
        y = np.random.randint(0, 4, size=(10,))
        dummy_blocks.append((X, y, None))
        
    esn = DeepESN(input_dim=128, n_layers=2, res_dim=100)
    esn.fit(dummy_blocks)
    
    preds = esn.predict(dummy_blocks)
    print(f"Prediction Output Shape (Block 0): {preds[0].shape}")
    print("Test passed.")
