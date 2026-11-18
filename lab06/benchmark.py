import time
import torch
from torch.utils.data import DataLoader, TensorDataset

def benchmark_workers():
    print("=== Benchmarking num_workers ===")
    X = torch.randn(2000, 1, 28, 28)
    y = torch.randint(0, 10, (2000,))
    dataset = TensorDataset(X, y)
    
    worker_options = [0, 1, 2, 4, 8]
    for w in worker_options:
        loader = DataLoader(dataset, batch_size=64, num_workers=w)
        start = time.time()
        for _ in loader:
            pass
        duration = time.time() - start
        print(f"num_workers={w}: {duration:.4f} sec")

def benchmark_batch_sizes():
    print("\n=== Benchmarking batch_sizes ===")
    X = torch.randn(5000, 1, 28, 28)
    y = torch.randint(0, 10, (5000,))
    dataset = TensorDataset(X, y)
    
    batch_sizes = [32, 64, 128, 256, 512]
    for bs in batch_sizes:
        loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=2)
        start = time.time()
        samples = 0
        for inputs, _ in loader:
            samples += inputs.size(0)
        duration = time.time() - start
        throughput = samples / duration
        print(f"Batch Size {bs:3d} | Throughput: {throughput:.2f} samples/sec")

if __name__ == "__main__":
    benchmark_workers()
    benchmark_batch_sizes()