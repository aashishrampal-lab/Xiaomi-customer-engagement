import torch
import time
import numpy as np

def benchmark_torch_argsort():
    # Check for GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    else:
        print("WARNING: CUDA not found. Benchmarking on CPU (will be slow).")
        device = torch.device("cpu")
        device_name = "CPU"

    print(f"Running PyTorch Benchmark on: {device_name}")

    # Configuration
    sizes = [10_000, 100_000, 1_000_000, 10_000_000, 50_000_000]
    iterations = 20
    warmup = 5

    print(f"{'Array Size':<15} | {'Mean Time (ms)':<15} | {'Std Dev (ms)':<15}")
    print("-" * 50)

    for n in sizes:
        # 1. Data Generation (Not timed)
        # Generate random int32 on the GPU directly
        arr = torch.randint(0, 1_000_000, (n,), device=device, dtype=torch.int32)
        
        # 2. Warmup
        for _ in range(warmup):
            _ = torch.argsort(arr)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # 3. Precise Timing Loop
        timings = []
        for _ in range(iterations):
            
            # CRITICAL: Wait for all previous GPU ops to finish before starting timer
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_t = time.perf_counter()
            
            # The operation to benchmark
            _ = torch.argsort(arr)
            
            # CRITICAL: Wait for argsort to actually finish
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_t = time.perf_counter()
            timings.append((end_t - start_t) * 1000)

        mean_t = np.mean(timings)
        std_t = np.std(timings)

        print(f"{n:<15,} | {mean_t:<15.4f} | {std_t:<15.4f}")

if __name__ == "__main__":
    benchmark_torch_argsort()