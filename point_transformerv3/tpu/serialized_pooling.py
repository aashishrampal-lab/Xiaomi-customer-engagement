import os

# --- 0. Environment Setup for SparseCore ---
# MUST be set before importing jax.
# These flags force the compiler to move gather/scatter ops to the SparseCore.
os.environ["LIBTPU_INIT_ARGS"] = (
    " --xla_tpu_enable_offloading_gather_to_sparsecore=true"
    " --xla_tpu_enable_offloading_scatter_to_sparsecore=true"
    " --xla_tpu_enable_concurrent_sparse_core_offloading=true"
)

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.tree_util import register_pytree_node
import math
import time
from typing import Any, Callable, Optional

# --- 1. Helper Classes with JAX Registration ---

class Point(dict):
    """
    Minimal wrapper to mimic Point behavior.
    """
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"'Point' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value

    def keys(self):
        return super().keys()

    def sparsify(self):
        pass

# Register Point as a Pytree so JAX can handle it
def flatten_point(point):
    """Splits Point into (children, metadata) for JAX."""
    keys = sorted(point.keys())
    children = [point[k] for k in keys]
    return children, keys

def unflatten_point(keys, children):
    """Reconstructs Point from (children, metadata)."""
    return Point(zip(keys, children))

register_pytree_node(Point, flatten_point, unflatten_point)


# --- 2. SerializedPooling with Inverse & Shuffle ---

class SerializedPooling(nn.Module):
    out_channels: int
    max_points: int  # REQUIRED for JIT: Worst case size
    stride: int = 2
    reduce: str = "max"
    shuffle_orders: bool = True
    traceable: bool = True

    @nn.compact
    def __call__(self, point: Point, training: bool = False):
        # Constants
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        
        # 1. Projection
        feat_projected = nn.Dense(self.out_channels)(point.feat)

        # 2. Downsample Code
        code_raw = point.serialized_code
        code = code_raw >> (pooling_depth * 3)
        
        # 3. Unique / Clustering (JIT COMPATIBLE)
        # We must specify 'size' to keep shapes static
        unique_code, cluster_ids, counts = jnp.unique(
            code[0],
            return_inverse=True,
            return_counts=True,
            size=self.max_points, 
            fill_value=0
        )
        
        # 4. Sorting and Indices
        indices = jnp.argsort(cluster_ids)
        
        # 5. Head Indices
        idx_ptr = jnp.concatenate([jnp.array([0]), jnp.cumsum(counts)])
        
        # Clamp to avoid index out of bounds on the padding
        safe_ptr = jnp.clip(idx_ptr[:-1], 0, self.max_points - 1)
        head_indices = indices[safe_ptr]

        # 6. Feature Aggregation (Segment Reduce)
        num_segments = self.max_points
        
        if self.reduce == "max":
            feat_agg = jax.ops.segment_max(feat_projected, cluster_ids, num_segments=num_segments)
            feat_agg = jnp.nan_to_num(feat_agg, neginf=0.0)
        elif self.reduce == "mean":
            feat_sum = jax.ops.segment_sum(feat_projected, cluster_ids, num_segments=num_segments)
            safe_counts = jnp.maximum(counts[:, None], 1) 
            feat_agg = feat_sum / safe_counts
        elif self.reduce == "sum":
            feat_agg = jax.ops.segment_sum(feat_projected, cluster_ids, num_segments=num_segments)

        # 7. Coord Aggregation
        # Note: Input coords are now padded to 8 channels, so output will also be 8 channels.
        coord_sum = jax.ops.segment_sum(point.coord, cluster_ids, num_segments=num_segments)
        safe_counts = jnp.maximum(counts[:, None], 1)
        coord_agg = coord_sum / safe_counts

        # 8. Downsample Code & Order & Inverse (UPDATED)
        code_down = code[:, head_indices]
        order = jnp.argsort(code_down, axis=-1)
        
        # --- Inverse Permutation Logic ---
        # "inverse.scatter_(dim=1, index=order, src=arange)" logic in JAX
        B, N_down = code_down.shape
        batch_idx = jnp.arange(B)[:, None]
        src = jnp.tile(jnp.arange(N_down), (B, 1))
        
        # inverse[b, order[b, i]] = i
        inverse = jnp.zeros_like(order).at[batch_idx, order].set(src)

        # 9. Shuffle Orders (Apples-to-Apples with PyTorch)
        if self.shuffle_orders:
            # We need a key for random permutation. 
            # In a rigorous setup, pass rng in variables. For benchmark, we make one.
            rng = self.make_rng('pooling') 
            perm = jax.random.permutation(rng, N_down)
            
            code_down = code_down[:, perm]
            order = order[:, perm]
            inverse = inverse[:, perm]

        # 10. Norm and Act
        feat_agg = nn.BatchNorm(use_running_average=not training)(feat_agg)
        feat_agg = nn.relu(feat_agg)

        # 11. Re-pack
        point_dict = {
            "feat": feat_agg,
            "coord": coord_agg, # This will be (N, 8)
            "serialized_code": code_down,
            "serialized_order": order,
            "serialized_inverse": inverse,  # Added to output
            "serialized_depth": point.serialized_depth - pooling_depth,
            "batch": point.batch[head_indices],
            "grid_coord": point.grid_coord[head_indices] >> pooling_depth # This will be (N, 8)
        }
        
        return Point(point_dict)

# --- 3. Benchmarking Script ---

def benchmark():
    # Use the first available device (TPU or GPU)
    try:
        device = jax.devices()[0]
        print(f"Benchmarking on device: {device}")
    except IndexError:
        print("No JAX devices found! Running on CPU?")

    num_points = 200_000 
    in_channels = 64
    out_channels = 128

    # Initialize Model
    model = SerializedPooling(
        out_channels=out_channels, 
        max_points=400000, 
        stride=2,
        shuffle_orders=True 
    )

    # Initialize Variables
    key = jax.random.PRNGKey(0)
    input_key, init_key, drop_key = jax.random.split(key, 3)
    
    # --- Helper to pad (N, 3) -> (N, 8) for SparseCore ---
    def pad_to_8(x):
        # Pads the last dimension: (0,0) for dim0, (0,5) for dim1
        return jnp.pad(x, ((0, 0), (0, 5)))

    # Data Generation
    # We pad 'coord' and 'grid_coord' because SparseCore requires innermost dim % 8 == 0
    dummy_input = Point({
        "feat": jax.random.normal(input_key, (num_points, in_channels)),
        "coord": pad_to_8(jax.random.normal(input_key, (num_points, 3))), 
        "grid_coord": pad_to_8(jax.random.randint(input_key, (num_points, 3), 0, 1000)),
        "serialized_code": jax.random.randint(input_key, (1, num_points), 0, 2**30),
        "serialized_depth": 10,
        "batch": jnp.zeros((num_points,), dtype=jnp.int32)
    })

    print("Initializing model...")
    # 'pooling' RNG collection is needed if shuffle_orders=True
    variables = model.init({'params': init_key, 'pooling': drop_key}, dummy_input)
    
    # JIT Compile
    # We must pass a fresh RNG key for the 'pooling' collection every time
    @jax.jit
    def forward_fn(vars, x, rng_key):
        return model.apply(vars, x, training=False, rngs={'pooling': rng_key})

    print("Warming up (compiling)...")
    _ = forward_fn(variables, dummy_input, drop_key).feat.block_until_ready()
    
    print("Running benchmark...")
    iterations = 10
    
    # We generate enough keys for the loop
    rng_keys = jax.random.split(drop_key, iterations)
    
    start_time = time.time()
    
    # Profiler is optional, useful for verification
     logdir = "gs://aashishrampaldev/tensorboard/point_transformer_v3/"
     jax.profiler.start_trace(logdir)
     print("JAX Profiling started...")
    
    for i in range(iterations):
        # We dispatch all iterations to the accelerator
        out = forward_fn(variables, dummy_input, rng_keys[i])
    
    # Blocking on the LAST output ensures we wait for the entire queue to finish
    out.feat.block_until_ready()
    
    end_time = time.time()
     jax.profiler.stop_trace()
    
    total_time_ms = (end_time - start_time) * 1000
    avg_time_ms = total_time_ms / iterations

    print("-" * 30)
    print(f"Total time ({iterations} runs): {total_time_ms:.2f} ms")
    print(f"Average time per forward pass: {avg_time_ms:.4f} ms")
    print("-" * 30)

if __name__ == "__main__":
    benchmark()
