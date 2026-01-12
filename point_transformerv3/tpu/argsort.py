import jax
import jax.numpy as jnp
import time
import numpy as np

def benchmark_jax_argsort():
    # Detect device
    device = jax.devices()[0]
    print(f"Running JAX Benchmark on: {device.platform.upper()} ({device.device_kind})")
    
    # Configuration
    sizes = [10_000, 100_000, 1_000_000, 10_000_000, 50_000_000]
    iterations = 20
    warmup = 5
    
    results = []

    print(f"{'Array Size':<15} | {'Mean Time (ms)':<15} | {'Std Dev (ms)':<15}")
    print("-" * 50)

    key = jax.random.PRNGKey(0)

    for n in sizes:
        # 1. Data Generation (Not timed)
        key, subkey = jax.random.split(key)
        # Generate random 32-bit integers
        arr = jax.random.randint(subkey, (n,), 0, 1_000_000, dtype=jnp.int32)
        
        # Force execution so data is actually on device before we start
        arr.block_until_ready()

        # 2. Warmup (Compile the kernel)
        for _ in range(warmup):
            _ = jnp.argsort(arr).block_until_ready()

        # 3. Precise Timing Loop
        timings = []
        for _ in range(iterations):
            # No need for explicit sync before start in JAX if previous op is blocked,
            # but good practice to be sure CPU is ready.
            start_t = time.perf_counter()
            
            # The operation to benchmark
            res = jnp.argsort(arr)
            
            # CRITICAL: Block until the GPU/TPU actually finishes writing 'res'
            res.block_until_ready()
            
            end_t = time.perf_counter()
            timings.append((end_t - start_t) * 1000) # Convert to ms

        mean_t = np.mean(timings)
        std_t = np.std(timings)
        
        print(f"{n:<15,} | {mean_t:<15.4f} | {std_t:<15.4f}")

if __name__ == "__main__":
    try:
        benchmark_jax_argsort()
    except Exception as e:
        print(f"\nError: {e}")
        print("Note: If using TPU, ensure you have allocated resources correctly.")