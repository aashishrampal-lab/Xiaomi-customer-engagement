import torch
import torch.nn as nn
import math
import time

# --- 1. Robust Import & Fallback ---
try:
    from torch_scatter import segment_csr
    print("[INFO] Successfully imported 'segment_csr' from torch_scatter.")
except (OSError, ImportError) as e:
    print(f"[WARN] torch_scatter import failed ({e}).")
    print("[INFO] Falling back to pure PyTorch 'segment_csr_torch' implementation.")
    
    def segment_csr(src, indptr, reduce="mean"):
        """
        Pure PyTorch implementation of torch_scatter.segment_csr.
        """
        if reduce == "min": reduce = "amin"
        if reduce == "max": reduce = "amax"
        
        # Calculate segment lengths
        counts = indptr[1:] - indptr[:-1]
        
        # Create index mapping for scatter
        index = torch.repeat_interleave(
            torch.arange(len(counts), device=src.device), 
            counts
        )
        
        # Initialize output buffer
        out_shape = (len(counts), src.shape[1])
        out = torch.zeros(out_shape, device=src.device, dtype=src.dtype)
        
        # Handle reduction
        out.scatter_reduce_(0, index.unsqueeze(1).expand_as(src), src, reduce=reduce, include_self=False)
        return out

# --- 2. Mock Classes ---

class Point(dict):
    """
    A wrapper around a dictionary to mimic the Point class.
    """
    def __init__(self, *args, **kwargs):
        super(Point, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def sparsify(self):
        pass

# --- 3. SerializedPooling Class ---

class SerializedPooling(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        
        self.norm = None
        if norm_layer is not None:
            self.norm = nn.Sequential(norm_layer(out_channels))
            
        self.act = None
        if act_layer is not None:
            self.act = nn.Sequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
            
        required_keys = {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }
        assert required_keys.issubset(point.keys()), "Run point.serialization() first"

        # Assume serialized_code is [1, N]
        code = point.serialized_code >> pooling_depth * 3
        
        # Unique finds clusters
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        
        _, indices = torch.sort(cluster)
        
        # index pointer for sorted point (CSR format)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        
        head_indices = indices[idx_ptr[:-1]]
        
        code = code[:, head_indices]
        order = torch.argsort(code)
        
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0], device=code.device)
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # Project features
        feat_projected = self.proj(point.feat)
        
        # Gather features
        feat_sorted = feat_projected[indices]
        coord_sorted = point.coord[indices]

        # --- USING segment_csr (either from lib or fallback) ---
        feat_pooled = segment_csr(feat_sorted, idx_ptr, reduce=self.reduce)
        coord_pooled = segment_csr(coord_sorted, idx_ptr, reduce="mean")
        # -------------------------------------------------------

        point_dict = Point(
            feat=feat_pooled,
            coord=coord_pooled,
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
            
        point = point_dict
        
        if self.norm is not None:
            point.feat = self.norm(point.feat)
        if self.act is not None:
            point.feat = self.act(point.feat)
            
        point.sparsify()
        return point


# --- 4. Benchmarking Script ---

def benchmark():
    # Configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Benchmarking on: {torch.cuda.get_device_name(device)}")
    else:
        print("CUDA not available. Benchmarking on CPU (Warning: Slow)")
        device = torch.device("cpu")
    
    num_points = 200_000
    in_channels = 64
    out_channels = 128
    
    # Model Setup
    model = SerializedPooling(
        in_channels=in_channels,
        out_channels=out_channels,
        stride=2,
        reduce="max",
        norm_layer=nn.BatchNorm1d,
        act_layer=nn.ReLU
    ).to(device)
    model.eval()

    # Data Generation
    print("Generating data...")
    serialized_code = torch.randint(0, 2**30, (1, num_points), device=device, dtype=torch.int64)

    input_point = Point(
        feat = torch.randn(num_points, in_channels, device=device, dtype=torch.float32),
        coord = torch.randn(num_points, 3, device=device, dtype=torch.float32),
        grid_coord = torch.randint(0, 1000, (num_points, 3), device=device, dtype=torch.int64),
        serialized_code = serialized_code,
        serialized_order = torch.argsort(serialized_code),
        serialized_inverse = torch.zeros((1, num_points), device=device, dtype=torch.int64),
        serialized_depth = 10,
        batch = torch.zeros(num_points, device=device, dtype=torch.int64)
    )

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_point)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmarking
    print("Running benchmark...")
    iterations = 100
    
    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_time = time.time()

    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_point)

    if device.type == 'cuda':
        end_event.record()
        torch.cuda.synchronize()
        total_time_ms = start_event.elapsed_time(end_event)
    else:
        total_time_ms = (time.time() - start_time) * 1000
    
    avg_time_ms = total_time_ms / iterations
    
    print("-" * 30)
    print(f"Total time ({iterations} runs): {total_time_ms:.2f} ms")
    print(f"Average time per forward pass: {avg_time_ms:.4f} ms")
    print("-" * 30)

if __name__ == "__main__":
    benchmark()