from datetime import datetime
import os
import time
from typing import Optional
from flax import linen as nn
import jax
import jax.numpy as jnp
import jax.profiler  # Import for Xprof
import numpy as np

from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel


# --- Configuration ---
MAX_CAPACITY = 100000
CHANNELS = 64
NUM_HEADS = 4
PATCH_SIZE = 1024
DTYPE = jnp.float32

class SerializedAttentionJax(nn.Module):
  channels: int
  num_heads: int
  patch_size: int
  qkv_bias: bool = True
  qk_scale: Optional[float] = None
  attn_drop: float = 0.0
  proj_drop: float = 0.0
  enable_rpe: bool = False
  dtype: jnp.dtype = DTYPE

  def setup(self):
    self.head_dim = self.channels // self.num_heads
    self.scale = self.qk_scale or (self.head_dim**-0.5)
    self.qkv = nn.Dense(self.channels * 3, use_bias=self.qkv_bias, dtype=self.dtype)
    self.proj = nn.Dense(self.channels, dtype=self.dtype)

    if self.enable_rpe:
      self.rpe_mlp = nn.Sequential(
          [nn.Dense(self.channels // 2, dtype=self.dtype), nn.relu, nn.Dense(self.num_heads, dtype=self.dtype)]
      )

  def __call__(self, x, grid_coords, batch_idx, deterministic=True):
    N, C = x.shape
    remainder = N % self.patch_size
    pad_len = (self.patch_size - remainder) % self.patch_size
    
    # 1. Padding so that we can evenly split the sequence into patches of size self.patch_size
    if pad_len > 0:
      x = jnp.pad(x, ((0, pad_len), (0, 0)))
      grid_coords = jnp.pad(grid_coords, ((0, pad_len), (0, 0)))
      batch_idx = jnp.pad(batch_idx, ((0, pad_len)), constant_values=-2)
    
    num_patches = x.shape[0] // self.patch_size
    x_patches = x.reshape(num_patches, self.patch_size, C)
    batch_idx_patches = batch_idx.reshape(num_patches, self.patch_size)

    # 2. QKV Projection & Reshape
    qkv = self.qkv(x_patches)
    # Reshape to (Batch/NumPatches, Seq/PatchSize, 3, Heads, Dim)
    qkv = qkv.reshape(
        num_patches, self.patch_size, 3, self.num_heads, self.head_dim
    )
    
    # Transpose for Splash Attention: (Batch, Seq, Head, Dim)
    # We want: (3, Batch, Seq, Heads, Dim)
    qkv = qkv.transpose(2, 0, 1, 3, 4)
    # qkv shape: (3, num_patches, patch_size, num_heads, self.head_dim)

    target_dim = max(self.head_dim, 128)
    pad_len = target_dim - self.head_dim

    if pad_len > 0:
        # Pad the last dimension of the combined qkv tensor
        pad_width = ((0, 0), (0, 0), (0, 0), (0, 0), (0, pad_len))
        qkv = jnp.pad(qkv, pad_width)
        # qkv shape: (3, num_patches, patch_size, num_heads, target_dim)

    q, k, v = qkv[0], qkv[1], qkv[2]

    # 3. Create Splash Mask
    # Create boolean mask: (Batch, 1, Seq, Seq)
    # True (1) = Keep/Attend, False (0) = Mask
    batch_mask = batch_idx_patches[:, :, None] == batch_idx_patches[:, None, :]

    # 4. Initialize Splash Kernel
    # We create the kernel here because the mask is dynamic (depends on batch_idx input).
    # Splash internally handles the complexity of blocking this mask.
    splash_kernel = splash_attention_kernel.make_splash_mha_single_device(
        mask=batch_mask,
    )

    # 5. Execute Attention
    # Note: RPE is skipped here because splash_attention fuses dot-product+softmax
    # and does not expose an interface to inject bias terms (like RPE) in between.
    attn_out = jax.vmap(splash_kernel)(
        q, k, v, 
        segment_ids=None 
    )[..., : self.head_dim]
    
    # attn_out shape is (num_patches, patch_size, num_heads, head_dim)

    # 6. Projection and Reshape Back
    # Flatten: (num_patches, patch_size, num_heads, head_dim) -> (num_patches, patch_size, C)
    attn_out_flat = attn_out.reshape(num_patches, self.patch_size, C)
    
    x_out = self.proj(attn_out_flat)
    
    # Remove padding and flatten back to (N, C)
    x_out = x_out.reshape(-1, C)[:N, :]
    # print(x_out.shape)
    # print(x.shape)
    
    return x_out


def generate_ragged_batch(max_capacity):
  lengths = []
  current_total = 0
  while current_total < max_capacity:
    l = np.random.randint(low=100, high=20000)
    if current_total + l > max_capacity:
      l = max_capacity - current_total
    if l <= 0:
      break
    lengths.append(l)
    current_total += l
  pad_total = max_capacity - current_total
  # print(f"Generated {len(lengths)} variable batches: {lengths}")
  # print(f"Buffer utilization: {current_total}/{max_capacity} points. Padding: {pad_total}")
  batch_ids_list = [
      np.full((length,), i, dtype=np.int32) for i, length in enumerate(lengths)
  ]
  if pad_total > 0:
    batch_ids_list.append(np.full((pad_total,), -1, dtype=np.int32))
  return jnp.array(np.concatenate(batch_ids_list))


def benchmark_variable_lengths():
  print(f"--- Benchmarking JAX Variable-Length Batches ---")
  print(f"Fixed Buffer Capacity: {MAX_CAPACITY}, Patch: {PATCH_SIZE}")

  model = SerializedAttentionJax(
      CHANNELS, NUM_HEADS, PATCH_SIZE, enable_rpe=False
  )
  key = jax.random.PRNGKey(42)
  dummy_feat = jax.random.normal(key, (MAX_CAPACITY, CHANNELS), dtype=DTYPE)
  dummy_coords = jax.random.randint(key, (MAX_CAPACITY, 3), 0, 1000)
  batch_idx = generate_ragged_batch(MAX_CAPACITY)
  variables = model.init(key, dummy_feat, dummy_coords, batch_idx)

  @jax.jit
  def forward_pass(vars, x, c, b):
    return model.apply(vars, x, c, b, deterministic=True)

  warmup_steps = 10
  print("Running warm-up steps...")
  start_warmup = time.time()
  for i in range(warmup_steps):
      batch_idx_warmup = generate_ragged_batch(MAX_CAPACITY)
      _ = forward_pass(variables, dummy_feat, dummy_coords, batch_idx_warmup).block_until_ready()
  print(f"Warm-up finished in {time.time() - start_warmup:.4f}s")

  print("\nRunning Benchmark with DIFFERENT batch configurations per step...")
  iterations = 20

  # --- XPROF Configuration ---
  gcs_bucket = "gs://aman-seervi-bucket"
  profile_base_dir = f"{gcs_bucket}/xprof_jax_benchmark"
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  log_dir = f"{profile_base_dir}/{timestamp}"
  # ---

  print(f"Starting Xprof trace, saving to: {log_dir}")
  jax.profiler.start_trace(log_dir)

  batch_idx = [generate_ragged_batch(MAX_CAPACITY) for _ in range(iterations)]
  print(f"Generated {len(batch_idx)} batches of size {MAX_CAPACITY}")
  start_time = time.time()
  for i in range(iterations):
    with jax.profiler.StepTraceAnnotation("train", step_num=i):
        out = forward_pass(variables, dummy_feat, dummy_coords, batch_idx[i])
        out.block_until_ready()
  
  total_time = time.time() - start_time

  avg_time = ((total_time) / iterations) * 1000

  print("Stopping Xprof trace...")
  jax.profiler.stop_trace()
  print(f"Xprof trace save process initiated to: {log_dir}")
  print(f"Check the GCS bucket. To view, you can use c2xprof or TensorBoard.")
  print(f"Example c2xprof command (run from Cloudtop):")
  print(
      "/google/src/head/depot/google3/cloud/tpu/tools/c2xprof/bin/c2xprof.par"
      f" --gcs_path={log_dir}"
  )
  # ---

  print(f"\nAverage Time per Step: {avg_time:.4f} ms")


if __name__ == "__main__":
  benchmark_variable_lengths()