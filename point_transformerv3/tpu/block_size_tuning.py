from datetime import datetime
import itertools
import json
import os
import time
from typing import Optional

from flax import linen as nn
import jax
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel import BlockSizes
import jax.numpy as jnp
import jax.profiler  # Import for Xprof
import numpy as np


# --- Configuration ---
MAX_CAPACITY = 100000
CHANNELS = 64
NUM_HEADS = 4
PATCH_SIZE = 1024
DTYPE = jnp.bfloat16


class SerializedAttentionJax(nn.Module):
  channels: int
  num_heads: int
  patch_size: int
  qkv_bias: bool = True
  qk_scale: Optional[float] = None
  attn_drop: float = 0.0
  proj_drop: float = 0.0
  enable_rpe: bool = False
  block_sizes: Optional[BlockSizes] = None
  dtype: jnp.dtype = DTYPE

  def setup(self):
    self.head_dim = self.channels // self.num_heads
    self.scale = self.qk_scale or (self.head_dim**-0.5)
    self.qkv = nn.Dense(
        self.channels * 3, use_bias=self.qkv_bias, dtype=self.dtype
    )
    self.proj = nn.Dense(self.channels, dtype=self.dtype)

    if self.enable_rpe:
      self.rpe_mlp = nn.Sequential([
          nn.Dense(self.channels // 2, dtype=self.dtype),
          nn.relu,
          nn.Dense(self.num_heads, dtype=self.dtype),
      ])

  def __call__(self, x, grid_coords, batch_idx, deterministic=True):
    N_orig, C = x.shape
    remainder = N_orig % self.patch_size
    pad_len = (self.patch_size - remainder) % self.patch_size

    if pad_len > 0:
      x = jnp.pad(x, ((0, pad_len), (0, 0)))
      grid_coords = jnp.pad(grid_coords, ((0, pad_len), (0, 0)))
      batch_idx = jnp.pad(batch_idx, ((0, pad_len)), constant_values=-2)

    N_padded = x.shape[0]
    num_patches = N_padded // self.patch_size
    x_patches = x.reshape(num_patches, self.patch_size, C)
    batch_idx_patches = batch_idx.reshape(num_patches, self.patch_size)

    qkv = self.qkv(x_patches)
    qkv = qkv.reshape(
        num_patches, self.patch_size, 3, self.num_heads, self.head_dim
    )
    qkv = qkv.transpose(2, 0, 1, 3, 4)

    target_dim = max(self.head_dim, 128)
    pad_len = target_dim - self.head_dim

    if pad_len > 0:
      # Pad the last dimension of the combined qkv tensor
      pad_width = ((0, 0), (0, 0), (0, 0), (0, 0), (0, pad_len))
      qkv = jnp.pad(qkv, pad_width)
      # qkv shape: (3, num_patches, patch_size, num_heads, target_dim)

    q, k, v = qkv[0], qkv[1], qkv[2]

    batch_mask = batch_idx_patches[:, :, None] == batch_idx_patches[:, None, :]

    # Use the block_sizes passed to the module
    splash_kernel = splash_attention_kernel.make_splash_mha_single_device(
        mask=batch_mask,
        block_sizes=self.block_sizes,
    )

    attn_out = jax.vmap(splash_kernel)(q, k, v, segment_ids=None)[
        ..., : self.head_dim
    ]

    attn_out_flat = attn_out.reshape(num_patches, self.patch_size, C)
    x_out = self.proj(attn_out_flat)
    x_out = x_out.reshape(-1, C)[:N_orig, :]
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
  batch_ids_list = [
      np.full((length,), i, dtype=np.int32) for i, length in enumerate(lengths)
  ]
  if pad_total > 0:
    batch_ids_list.append(np.full((pad_total,), -1, dtype=np.int32))
  return jnp.array(np.concatenate(batch_ids_list))


def benchmark_block_size_tuning():
  print(f"--- Benchmarking JAX Splash Attention Block Size Tuning ---")
  print(f"Fixed Buffer Capacity: {MAX_CAPACITY}, Patch: {PATCH_SIZE}")
  print(
      f"CHANNELS: {CHANNELS}, NUM_HEADS: {NUM_HEADS}, Head Dim:"
      f" {CHANNELS // NUM_HEADS}"
  )
  print(
      f"Sequence Length per patch: {PATCH_SIZE}, Padded Head Dim:"
      f" {max(CHANNELS // NUM_HEADS, 128)}"
  )

  key = jax.random.PRNGKey(42)
  dummy_feat = jax.random.normal(key, (MAX_CAPACITY, CHANNELS), dtype=DTYPE)
  dummy_coords = jax.random.randint(key, (MAX_CAPACITY, 3), 0, 1000)
  batch_idx = generate_ragged_batch(MAX_CAPACITY)

  # --- Define BlockSizes configurations to test ---
  block_options = [128, 256, 512, 1024]
  block_sizes_options = [None]  # Test kernel defaults

  # Add combinations for block_q and block_kv
  for bq, bkv in itertools.product(block_options, block_options):
    # Example: Tune only block_q and block_kv, keep others default within BlockSizes.
    block_sizes_options.append(BlockSizes(block_q=bq, block_kv=bkv))

  results = {}

  for i, block_sizes in enumerate(block_sizes_options):
    print(f"\n--- Testing Configuration {i+1}/{len(block_sizes_options)} ---")
    if block_sizes is None:
      print("BlockSizes: Kernel Defaults (None)")
    else:
      print(f"BlockSizes: {block_sizes}")

    model = SerializedAttentionJax(
        CHANNELS,
        NUM_HEADS,
        PATCH_SIZE,
        enable_rpe=False,
        block_sizes=block_sizes,
    )

    # --- XPROF Configuration ---
    gcs_bucket = "gs://aman-seervi-bucket"
    profile_base_dir = f"{gcs_bucket}/xprof_jax_benchmark"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"{profile_base_dir}/{timestamp}"
    # ---

    try:
      variables = model.init(key, dummy_feat, dummy_coords, batch_idx)

      @jax.jit
      def forward_pass(vars, x, c, b):
        return model.apply(vars, x, c, b, deterministic=True)

      # Warmup
      print("Compiling and Warmup...")
      compile_start_time = time.time()
      for _ in range(10):
        _ = forward_pass(
            variables, dummy_feat, dummy_coords, batch_idx
        ).block_until_ready()
      print(f"Compilation/Warmup took {time.time() - compile_start_time:.4f}s")

      print("Starting Xprof trace...")
      jax.profiler.start_trace(log_dir)

      # Timing
      iterations = 20
      start_time = time.time()
      for i in range(iterations):
        with jax.profiler.StepTraceAnnotation("train", step_num=i):
          out = forward_pass(variables, dummy_feat, dummy_coords, batch_idx)
          out.block_until_ready()
      end_time = time.time()

      avg_time_ms = ((end_time - start_time) / iterations) * 1000
      print(f"Average Time per Step: {avg_time_ms:.4f} ms")
      results[str(block_sizes)] = avg_time_ms

      print("Stopping Xprof trace...")
      jax.profiler.stop_trace()
      print(f"Xprof trace save process initiated to: {log_dir}")
      print(
          f"Check the GCS bucket. To view, you can use c2xprof or TensorBoard."
      )
      print(f"Example c2xprof command (run from Cloudtop):")
      print(
          "/google/src/head/depot/google3/cloud/tpu/tools/c2xprof/bin/c2xprof.par"
          f" --gcs_path={log_dir}"
      )
      # ---

    except Exception as e:
      print(f"Error with BlockSizes {block_sizes}: {e}")
      results[str(block_sizes)] = "Error"

  print("\n--- Benchmark Results Summary ---")
  for bs, time_ms in results.items():
    print(f"BlockSizes: {bs} -> {time_ms}")

  best_config = min(
      results,
      key=lambda k: results[k]
      if isinstance(results[k], float)
      else float("inf"),
  )
  print(f"\nBest Configuration: {best_config} -> {results[best_config]:.4f} ms")

  # -----------------------------------------------------------------------------
  #               Save xprof for best config
  # -----------------------------------------------------------------------------

  # --- XPROF Configuration ---
  gcs_bucket = "gs://aman-seervi-bucket"
  profile_base_dir = f"{gcs_bucket}/xprof_jax_benchmark"
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  log_dir = f"{profile_base_dir}/{timestamp}"
  # ---

  # Find the BlockSizes object in block_sizes_options that matches the best_config string
  best_block_sizes_obj = None
  for bs in block_sizes_options:
    if str(bs) == best_config:
      best_block_sizes_obj = bs
      break

  print(f"\n--- Running Xprof for Best Configuration: {best_config} ---")
  model = SerializedAttentionJax(
      CHANNELS,
      NUM_HEADS,
      PATCH_SIZE,
      enable_rpe=False,
      block_sizes=best_block_sizes_obj,  # Use the retrieved object
  )
  variables = model.init(key, dummy_feat, dummy_coords, batch_idx)

  @jax.jit
  def forward_pass(vars, x, c, b):
    return model.apply(vars, x, c, b, deterministic=True)

  # Warmup for Xprof
  print("Warmup for Xprof...")
  for _ in range(5):
    _ = forward_pass(
        variables, dummy_feat, dummy_coords, batch_idx
    ).block_until_ready()

  print("Starting Xprof trace...")
  jax.profiler.start_trace(log_dir)  # Start trace and point to the log_dir

  for i in range(20):  # Use iterations as defined before
    with jax.profiler.StepTraceAnnotation("train", step_num=i):
      _ = forward_pass(
          variables, dummy_feat, dummy_coords, batch_idx
      ).block_until_ready()

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


if __name__ == "__main__":
  benchmark_block_size_tuning()
