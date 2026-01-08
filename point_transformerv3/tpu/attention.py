from datetime import datetime
import os
import time
from typing import Optional
from flax import linen as nn
import jax
import jax.numpy as jnp
import jax.profiler  # Import for Xprof
import numpy as np

# --- Configuration ---
MAX_CAPACITY = 100000
CHANNELS = 64
NUM_HEADS = 4
PATCH_SIZE = 1024


class SerializedAttentionJax(nn.Module):
  channels: int
  num_heads: int
  patch_size: int
  qkv_bias: bool = True
  qk_scale: Optional[float] = None
  attn_drop: float = 0.0
  proj_drop: float = 0.0
  enable_rpe: bool = False

  def setup(self):
    self.head_dim = self.channels // self.num_heads
    self.scale = self.qk_scale or (self.head_dim**-0.5)
    self.qkv = nn.Dense(self.channels * 3, use_bias=self.qkv_bias)
    self.proj = nn.Dense(self.channels)
    if self.enable_rpe:
      self.rpe_mlp = nn.Sequential(
          [nn.Dense(self.channels // 2), nn.relu, nn.Dense(self.num_heads)]
      )

  def __call__(self, x, grid_coords, batch_idx, deterministic=True):
    N, C = x.shape
    remainder = N % self.patch_size
    pad_len = (self.patch_size - remainder) % self.patch_size
    if pad_len > 0:
      x = jnp.pad(x, ((0, pad_len), (0, 0)))
      grid_coords = jnp.pad(grid_coords, ((0, pad_len), (0, 0)))
      batch_idx = jnp.pad(batch_idx, ((0, pad_len)), constant_values=-2)
    num_patches = x.shape[0] // self.patch_size
    x_patches = x.reshape(num_patches, self.patch_size, C)
    batch_idx_patches = batch_idx.reshape(num_patches, self.patch_size)
    qkv = self.qkv(x_patches)
    qkv = qkv.reshape(
        num_patches, self.patch_size, 3, self.num_heads, self.head_dim
    )
    qkv = qkv.transpose(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn_logits = jnp.einsum("...qd,...kd->...qk", q, k) * self.scale
    batch_mask = batch_idx_patches[:, :, None] == batch_idx_patches[:, None, :]
    batch_mask = batch_mask[:, None, :, :]
    min_float = jnp.finfo(jnp.float32).min
    attn_logits = jnp.where(batch_mask, attn_logits, min_float)
    if self.enable_rpe:
      coords_patches = grid_coords.reshape(num_patches, self.patch_size, 3)
      rel_pos = coords_patches[:, :, None, :] - coords_patches[:, None, :, :]
      rpe_bias = self.rpe_mlp(rel_pos).transpose(0, 3, 1, 2)
      attn_logits = attn_logits + rpe_bias
    attn_weights = nn.softmax(attn_logits, axis=-1)
    attn_out = jnp.einsum("...qk,...kd->...qd", attn_weights, v)
    x_out = self.proj(
        attn_out.transpose(0, 2, 1, 3).reshape(num_patches, self.patch_size, C)
    )
    x_out = x_out.reshape(-1, C)[:N, :]
    return x[:N, :] + x_out


def generate_ragged_batch(max_capacity):
  lengths = []
  current_total = 0
  while current_total < max_capacity:
    l = np.random.randint(low=10000, high=30000)
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
  dummy_feat = jax.random.normal(key, (MAX_CAPACITY, CHANNELS))
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
  
  iterations = 20

  # --- XPROF Configuration ---
  gcs_bucket = "gs://aman-seervi-bucket"
  profile_base_dir = f"{gcs_bucket}/xprof_jax_benchmark"
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  log_dir = f"{profile_base_dir}/{timestamp}"
  # ---

  print(f"Starting Xprof trace, saving to: {log_dir}")
  jax.profiler.start_trace(log_dir)

  total_time=0
  for i in range(iterations):
    if i % 5 == 0:
      batch_idx = generate_ragged_batch(MAX_CAPACITY)
    total_time -= time.time()
    with jax.profiler.StepTraceAnnotation("train", step_num=i):
        out = forward_pass(variables, dummy_feat, dummy_coords, batch_idx)
        out.block_until_ready()
    total_time +=time.time()

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
  print(
      "Success: The model handled variable batch lengths without recompiling!"
  )



if __name__ == "__main__":
  benchmark_variable_lengths()
