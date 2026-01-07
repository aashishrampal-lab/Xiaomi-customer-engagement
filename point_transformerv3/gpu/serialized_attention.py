import argparse
import math
import time
import torch
import torch.nn as nn

# Check for Flash Attention
try:
  import flash_attn

  FLASH_ATTN_AVAILABLE = True
except ImportError:
  FLASH_ATTN_AVAILABLE = False
  print(
      "Warning: flash_attn not installed. Benchmarking generic PyTorch"
      " implementation."
  )

# ==========================================
# 1. Helper Classes & Functions
# ==========================================


class Point(dict):
  """Minimal Mock of the Point class from pointcept.models.utils.structure"""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    for k, v in kwargs.items():
      self.__setattr__(k, v)

  def __getattr__(self, key):
    try:
      return self[key]
    except KeyError:
      raise AttributeError(key)

  def __setattr__(self, key, value):
    self[key] = value


class PointModule(nn.Module):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)


def offset2bincount(offset):
  """Calculates the number of points in each batch element.

  offset: [n1, n1+n2, ..., N] returns: [n1, n2, ...]
  """
  return torch.cat([offset[0:1], offset[1:] - offset[:-1]])


# ==========================================
# 2. Model Definitions
# ==========================================


class RPE(torch.nn.Module):

  def __init__(self, patch_size, num_heads):
    super().__init__()
    self.patch_size = patch_size
    self.num_heads = num_heads
    self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
    self.rpe_num = 2 * self.pos_bnd + 1
    self.rpe_table = torch.nn.Parameter(
        torch.zeros(3 * self.rpe_num, num_heads)
    )
    torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

  def forward(self, coord):
    idx = (
        coord.clamp(-self.pos_bnd, self.pos_bnd)
        + self.pos_bnd
        + torch.arange(3, device=coord.device) * self.rpe_num
    )
    out = self.rpe_table.index_select(0, idx.reshape(-1))
    out = out.view(idx.shape + (-1,)).sum(3)
    out = out.permute(0, 3, 1, 2)
    return out


class SerializedAttention(PointModule):

  def __init__(
      self,
      channels,
      num_heads,
      patch_size,
      qkv_bias=True,
      qk_scale=None,
      attn_drop=0.0,
      proj_drop=0.0,
      order_index=0,
      enable_rpe=False,
      enable_flash=True,
      upcast_attention=True,
      upcast_softmax=True,
  ):
    super().__init__()
    assert channels % num_heads == 0
    self.channels = channels
    self.num_heads = num_heads
    self.scale = qk_scale or (channels // num_heads) ** -0.5
    self.order_index = order_index
    self.upcast_attention = upcast_attention
    self.upcast_softmax = upcast_softmax
    self.enable_rpe = enable_rpe
    self.enable_flash = enable_flash
    if enable_flash:
      assert (
          enable_rpe is False
      ), "Set enable_rpe to False when enable Flash Attention"
      assert (
          upcast_attention is False
      ), "Set upcast_attention to False when enable Flash Attention"
      assert (
          upcast_softmax is False
      ), "Set upcast_softmax to False when enable Flash Attention"
      assert flash_attn is not None, "Make sure flash_attn is installed."
      self.patch_size = patch_size
      self.attn_drop = attn_drop
    else:
      self.patch_size_max = patch_size
      self.patch_size = 0
      self.attn_drop = torch.nn.Dropout(attn_drop)

    self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
    self.proj = torch.nn.Linear(channels, channels)
    self.proj_drop = torch.nn.Dropout(proj_drop)
    self.softmax = torch.nn.Softmax(dim=-1)
    self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

  @torch.no_grad()
  def get_rel_pos(self, point, order):
    K = self.patch_size
    rel_pos_key = f"rel_pos_{self.order_index}"
    if rel_pos_key not in point.keys():
      grid_coord = point.grid_coord[order]
      grid_coord = grid_coord.reshape(-1, K, 3)
      point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
    return point[rel_pos_key]

  @torch.no_grad()
  def get_padding_and_inverse(self, point):
    pad_key = "pad"
    unpad_key = "unpad"
    cu_seqlens_key = "cu_seqlens_key"
    # We append patch_size to keys to allow dynamic changing of patch_size in benchmark loops
    # without key collision if re-using the same point object logic repeatedly.
    if (
        pad_key not in point.keys()
        or unpad_key not in point.keys()
        or cu_seqlens_key not in point.keys()
    ):
      offset = point.offset
      bincount = offset2bincount(offset)
      bincount_pad = (
          torch.div(
              bincount + self.patch_size - 1,
              self.patch_size,
              rounding_mode="trunc",
          )
          * self.patch_size
      )
      mask_pad = bincount > self.patch_size
      bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
      _offset = nn.functional.pad(offset, (1, 0))
      _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
      pad = torch.arange(_offset_pad[-1], device=offset.device)
      unpad = torch.arange(_offset[-1], device=offset.device)
      cu_seqlens = []
      for i in range(len(offset)):
        unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
        if bincount[i] != bincount_pad[i]:
          pad[
              _offset_pad[i + 1]
              - self.patch_size
              + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
          ] = pad[
              _offset_pad[i + 1]
              - 2 * self.patch_size
              + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
              - self.patch_size
          ]
        pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
        cu_seqlens.append(
            torch.arange(
                _offset_pad[i],
                _offset_pad[i + 1],
                step=self.patch_size,
                dtype=torch.int32,
                device=offset.device,
            )
        )
      point[pad_key] = pad
      point[unpad_key] = unpad
      point[cu_seqlens_key] = nn.functional.pad(
          torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
      )
    return point[pad_key], point[unpad_key], point[cu_seqlens_key]

  def forward(self, point):
    if not self.enable_flash:
      self.patch_size = min(
          offset2bincount(point.offset).min().tolist(), self.patch_size_max
      )

    H = self.num_heads
    K = self.patch_size
    C = self.channels

    pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

    if isinstance(point.serialized_order, list):
      order = point.serialized_order[self.order_index][pad]
      inverse = unpad[point.serialized_inverse[self.order_index]]
    else:
      order = point.serialized_order[pad]
      inverse = unpad[point.serialized_inverse]

    qkv = self.qkv(point.feat)[order]

    if not self.enable_flash:
      q, k, v = (
          qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
      )
      if self.upcast_attention:
        q = q.float()
        k = k.float()
      attn = (q * self.scale) @ k.transpose(-2, -1)
      if self.enable_rpe:
        attn = attn + self.rpe(self.get_rel_pos(point, order))
      if self.upcast_softmax:
        attn = attn.float()
      attn = self.softmax(attn)
      attn = self.attn_drop(attn).to(qkv.dtype)
      feat = (attn @ v).transpose(1, 2).reshape(-1, C)
    else:
      feat = flash_attn.flash_attn_varlen_qkvpacked_func(
          qkv.to(torch.bfloat16).reshape(-1, 3, H, C // H),
          cu_seqlens,
          max_seqlen=self.patch_size,
          dropout_p=self.attn_drop if self.training else 0,
          softmax_scale=self.scale,
      ).reshape(-1, C)
      feat = feat.to(qkv.dtype)

    feat = feat[inverse]
    feat = self.proj(feat)
    feat = self.proj_drop(feat)
    point.feat = feat
    return point


# ==========================================
# 3. Benchmark Logic
# ==========================================


def get_args():
  parser = argparse.ArgumentParser(
      description="Benchmark SerializedAttention on GPU"
  )

  # Required/Key arguments
  parser.add_argument(
      "--patch_size", type=int, default=48, help="Size of the attention patch"
  )
  parser.add_argument(
      "--points", type=int, default=100000, help="Total number of points"
  )
  parser.add_argument(
      "--channels", type=int, default=64, help="Number of channels"
  )
  parser.add_argument(
      "--heads", type=int, default=4, help="Number of attention heads"
  )
  parser.add_argument(
      "--batch_size", type=int, default=4, help="Batch size (simulated)"
  )
  parser.add_argument(
      "--iters", type=int, default=1000, help="Number of benchmark iterations"
  )

  return parser.parse_args()


def run_benchmark():
  args = get_args()

  # Assign from args
  PATCH_SIZE = args.patch_size
  POINTS = args.points
  CHANNELS = args.channels
  NUM_HEADS = args.heads
  BATCH_SIZE = args.batch_size

  DEVICE = (
      torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  )
  print(
      f"Benchmarking on: {DEVICE}"
      f" ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})"
  )
  print(
      f"Config: Patch Size={PATCH_SIZE}, Points={POINTS}, Channels={CHANNELS},"
      f" Heads={NUM_HEADS}, Batch={BATCH_SIZE}"
  )

  # --- Data Generation ---
  print(f"Generating input data...")

  feat = torch.randn(POINTS, CHANNELS, device=DEVICE, dtype=torch.float16)

  points_per_batch = POINTS // BATCH_SIZE
  # Ensure offset does not start with 0
  offset = torch.arange(
      points_per_batch,
      POINTS + 1,
      points_per_batch,
      device=DEVICE,
      dtype=torch.int32,
  )
  # Fix last element rounding issues if any
  if len(offset) > 0:
    offset[-1] = POINTS

  grid_coord = torch.randint(
      0, 1000, (POINTS, 3), device=DEVICE, dtype=torch.int32
  )
  rand_perm = torch.randperm(POINTS, device=DEVICE)
  inverse_perm = torch.argsort(rand_perm)

  input_point = Point(
      feat=feat,
      offset=offset,
      grid_coord=grid_coord,
      serialized_order=[rand_perm],
      serialized_inverse=[inverse_perm],
  )

  # --- Model Initialization ---
  model = SerializedAttention(
      channels=CHANNELS,
      num_heads=NUM_HEADS,
      patch_size=PATCH_SIZE,
      enable_flash=FLASH_ATTN_AVAILABLE,
      enable_rpe=False,
      upcast_attention=False,
      upcast_softmax=False,
  ).to(DEVICE)

  model.eval()
  model.half()  # FP16

  # --- Benchmarking ---
  warmup_iters = 100
  run_iters = args.iters

  print("Warming up...")
  with torch.no_grad():
    for _ in range(warmup_iters):
      _ = model(input_point)

  input_point.feat = feat.clone()  # Reset

  print(f"Running {run_iters} iterations...")
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)

  torch.cuda.synchronize()
  start_event.record()

  with torch.no_grad():
    for _ in range(run_iters):
      _ = model(input_point)

  end_event.record()
  torch.cuda.synchronize()

  elapsed_time_ms = start_event.elapsed_time(end_event)
  avg_time_ms = elapsed_time_ms / run_iters

  print(f"--------------------------------------------------")
  print(f"Results for Patch Size {PATCH_SIZE}:")
  print(f"Total time: {elapsed_time_ms:.2f} ms")
  print(f"Average time per forward pass: {avg_time_ms:.4f} ms")
  print(
      f"Throughput: {POINTS * run_iters / (elapsed_time_ms / 1000):.2f}"
      " points/sec"
  )
  print(f"--------------------------------------------------")


if __name__ == "__main__":
  run_benchmark()
