# ADAS Model Benchmarking: ResNet50 & Point Transformer v3

This repository contains scripts and code for benchmarking ResNet50 and key components of the Point Transformer v3 (PTv3) model, specifically for Advanced Driver Assistance Systems (ADAS) workloads. The benchmarks compare performance between GPU (using PyTorch) and Google TPU (using JAX).

## Objective

The primary goal is to evaluate the feasibility and performance of Google TPUs (v6e) against GPUs (H100) for training and inference tasks relevant to ADAS, encompassing both standard Convolutional Neural Networks and models designed for sparse point cloud data.

## Repository Structure

.
├── point_transformerv3
│ ├── gpu
│ │ ├── argsort.py
│ │ ├── grid_pooling.py
│ │ └── serialized_attention.py
│ └── tpu
│ ├── argsort.py
│ ├── attention.py
│ ├── block_size_tuning.py
│ ├── serialized_pooling.py
│ ├── splash_attention.py
│ └── splash_attention_pad_fusion.py
└── resnet50
├── jax_resnet
├── tests
├── LICENSE
├── requirements-dev.txt
├── run_benchmark.py
├── setup.cfg
└── setup.py



*   **`point_transformerv3`**: Contains benchmarks for specific kernels and operations within the Point Transformer v3 model.
    *   **`gpu`**: PyTorch implementations for GPU benchmarking.
        *   `argsort.py`: ArgSort benchmark.
        *   `grid_pooling.py`: Benchmark for pooling layers.
        *   `serialized_attention.py`: Benchmark for attention layers.
    *   **`tpu`**: JAX implementations for TPU benchmarking.
        *   `argsort.py`: ArgSort benchmark.
        *   `attention.py`: Basic attention implementation.
        *   `block_size_tuning.py`: Script for tuning block sizes in attention.
        *   `serialized_pooling.py`: Benchmark for pooling layers.
        *   `splash_attention.py`: Optimized Splash Attention implementation.
        *   `splash_attention_pad_fusion.py`: Splash Attention with fused padding.
*   **`resnet50`**: Contains code for benchmarking the ResNet50 model.
    *   `jax_resnet`: JAX implementation of ResNet50 for TPU.
    *   `run_benchmark.py`: Script to execute ResNet50 benchmarks.

## Models & Operations Benchmarked

1.  **ResNet50**: Standard image classification model. Benchmarks likely focus on throughput and latency.
2.  **Point Transformer v3 (PTv3) Components**:
    *   **Serialized Attention**: Benchmarking standard and optimized (Splash Attention) attention mechanisms on both GPU and TPU.
    *   **Pooling Layers**: Evaluating performance of grid/serialized pooling operations.
    *   **ArgSort**: Isolated benchmark of the ArgSort operation, crucial for sparse data processing.

## Usage

Each directory (`gpu`, `tpu` within `point_transformerv3`, and `resnet50`) contains specific scripts. To run the benchmarks, navigate to the respective directories and execute the Python scripts. Ensure you have the necessary environments (CUDA for GPU, TPU dependencies for JAX) and libraries installed (as per `requirements-dev.txt` for ResNet50).

Example (conceptual):

```bash
# Example for running a TPU attention benchmark
cd point_transformerv3/tpu
python attention.py

# Example for running ResNet50 benchmark
cd resnet50
python run_benchmark.py