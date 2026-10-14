import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

os.makedirs('results', exist_ok=True)



def prepare_data():
    digits = load_digits()
    X = digits.data / 16.0  
    y = digits.target
    y_onehot = np.eye(10)[y]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.30, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test


def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def compute_loss(Y, A3):
    eps = 1e-15
    A3_clipped = np.clip(A3, eps, 1 - eps)
    return -np.mean(np.sum(Y * np.log(A3_clipped), axis=1))

def init_params(W_dim):
    np.random.seed(42)
    return {
        'W1': np.random.randn(64, W_dim) * 0.01,
        'b1': np.zeros(W_dim),
        'W2': np.random.randn(W_dim, W_dim) * 0.01,
        'b2': np.zeros(W_dim),
        'W3': np.random.randn(W_dim, 10) * 0.01,
        'b3': np.zeros(10)
    }

def forward(X, params):
    cache = {'X': X}
    cache['Z1'] = X @ params['W1'] + params['b1']
    cache['A1'] = relu(cache['Z1'])
    cache['Z2'] = cache['A1'] @ params['W2'] + params['b2']
    cache['A2'] = relu(cache['Z2'])
    cache['Z3'] = cache['A2'] @ params['W3'] + params['b3']
    cache['A3'] = softmax(cache['Z3'])
    return cache['A3'], cache

def backward(Y, cache, params):
    N = Y.shape[0]
    grads = {}
    
    dZ3 = (cache['A3'] - Y) / N
    grads['dW3'] = cache['A2'].T @ dZ3
    grads['db3'] = np.sum(dZ3, axis=0)
    
    dA2 = dZ3 @ params['W3'].T
    dZ2 = dA2 * relu_derivative(cache['Z2'])
    grads['dW2'] = cache['A1'].T @ dZ2
    grads['db2'] = np.sum(dZ2, axis=0)
    
    dA1 = dZ2 @ params['W2'].T
    dZ1 = dA1 * relu_derivative(cache['Z1'])
    grads['dW1'] = cache['X'].T @ dZ1
    grads['db1'] = np.sum(dZ1, axis=0)
    
    return grads


def gradient_check():
    print("--- 1. Running Gradient Check ---")
    X_val = np.random.randn(20, 64)
    Y_val = np.eye(10)[np.random.choice(10, 20)]
    params = init_params(16)
    
    A3, cache = forward(X_val, params)
    grads = backward(Y_val, cache, params)
    
    keys = list(params.keys())
    passed = 0
    
    for _ in range(10):
        key = np.random.choice(keys)
        idx = tuple(np.random.randint(0, dim) for dim in params[key].shape)
        
        eps = 1e-5
        params[key][idx] += eps
        loss_plus, _ = forward(X_val, params)
        l_plus = compute_loss(Y_val, loss_plus)
        
        params[key][idx] -= 2 * eps
        loss_minus, _ = forward(X_val, params)
        l_minus = compute_loss(Y_val, loss_minus)
        
        params[key][idx] += eps
        
        num_grad = (l_plus - l_minus) / (2 * eps)
        ana_grad = grads['d' + key][idx]
        
        rel_error = np.abs(ana_grad - num_grad) / (np.abs(ana_grad) + np.abs(num_grad) + 1e-8)
        
        if rel_error < 1e-6:
            passed += 1
            
    print(f"Gradient Check passed for {passed}/10 tested weights.")
    assert passed >= 9, "Gradient check failed!"
    print("Gradient Check successful!\n")


def train_and_eval(X_train, y_train, X_test, y_test):
    print("--- 2. Training Models ---")
    learning_rates = [0.001, 0.01, 0.1]
    widths = [16, 64]
    
    plt.figure(figsize=(10, 6))
    
    for W_dim in widths:
        for lr in learning_rates:
            params = init_params(W_dim)
            loss_history = []
            
            for epoch in range(50):
                indices = np.arange(X_train.shape[0])
                np.random.shuffle(indices)
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]
                
                for i in range(0, X_train.shape[0], 32):
                    X_batch = X_shuffled[i:i+32]
                    y_batch = y_shuffled[i:i+32]
                    
                    A3, cache = forward(X_batch, params)
                    grads = backward(y_batch, cache, params)
                    
                    for k in params:
                        params[k] -= lr * grads['d' + k]
                        
                A3_full, _ = forward(X_train, params)
                loss_history.append(compute_loss(y_train, A3_full))
                
            label = f"W={W_dim}, lr={lr}"
            plt.plot(loss_history, label=label)
            
            A3_test, _ = forward(X_test, params)
            acc = np.mean(np.argmax(A3_test, axis=1) == np.argmax(y_test, axis=1))
            print(f"Config ({label}): Test Accuracy = {acc:.4f}")
            
    plt.title("Loss Curves for Different Configurations")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig('results/loss_curves.png')
    plt.close()
    print("Loss curves saved to 'results/loss_curves.png'\n")


def benchmark_timing(X_train, y_train):
    print("--- 3. Benchmarking Forward & Backward Timing ---")
    params = init_params(16)
    X_batch = X_train[:32]
    y_batch = y_train[:32]
    
    fwd_times = []
    bwd_times = []
    
    for _ in range(100):
        t0 = time.perf_counter()
        A3, cache = forward(X_batch, params)
        t1 = time.perf_counter()
        fwd_times.append(t1 - t0)
        
        t0 = time.perf_counter()
        _ = backward(y_batch, cache, params)
        t1 = time.perf_counter()
        bwd_times.append(t1 - t0)
        
    med_fwd = np.median(fwd_times) * 1000
    med_bwd = np.median(bwd_times) * 1000
    
    print(f"Forward Pass Median Time:  {med_fwd:.4f} ms")
    print(f"Backward Pass Median Time: {med_bwd:.4f} ms")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    gradient_check()
    train_and_eval(X_train, y_train, X_test, y_test)
    benchmark_timing(X_train, y_train)