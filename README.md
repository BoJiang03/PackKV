# PackKV

This is the official repository for the paper **"PackKV: Reducing KV Cache Memory Footprint through LLM-Aware Lossy Compression"** (IPDPS 2026).

PackKV is a high-performance framework designed to reduce the memory footprint of Key-Value (KV) cache for Large Language Models Inference with Lossy Compression. By utilizing custom CUDA kernels and sophisticated lossy compression techniques, PackKV aims to improve reduce memory usage and inference throughput.

## Installation

### Prerequisites

- Linux
- NVIDIA GPU with CUDA support
- Anaconda or Miniconda or Miniforge

### Step 1: Set up the Environment

Create a new Conda environment using the provided configuration file:

```bash
conda env create -f environment.yml
conda activate packkv_pub
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128 # to support RTX Pro 6000
pip install flash-attn==2.8.1 --no-build-isolation
pip install -r requirements.txt
```

### Step 2: Install CUDA Extensions

Compile and install the custom CUDA kernels required for PackKV:

```bash
cd packkv_cuda_ext
pip install -e .
cd ..
```

if you failed to compile this extension, you can try to modify the `setup.py` file:
```bash
'nvcc': [
    '-O3',
    # '-gencode=arch=compute_70,code=compute_70',
    # '-gencode=arch=compute_80,code=sm_80',
    # '-gencode=arch=compute_89,code=sm_89',
    '-gencode=arch=compute_120,code=sm_120', # this works for RTX Pro 6000
]
```

### Step 3: Test Run

```bash
cd scripts
PYTHONPATH=.. python ./rebuttal_throughout.py
cd ..
```

## Project Structure

- **`packkv_cuda_ext/`**: C++ and CUDA source code for the packkv custom extension.
- **`models/`**: implementations of LLMs used in PackKV Experiments.
- **`scripts/`**: Scripts for automation, benchmarking, and generating experimental results.
- **`evaluation/`**: Code related to model evaluation.
- **`utils/`**: Helper functions and utilities.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
