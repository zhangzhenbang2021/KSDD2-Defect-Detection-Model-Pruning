# KSDD2 Defect Detection Model Pruning

An end-to-end channel pruning pipeline for the KSDD2 (Kaggle Surface Defect Detection) defect detection model, featuring multiple pruning strategies including L1 norm, Network Slimming, hardware-aligned pruning, defect-aware pruning, activation correlation pruning, and Taylor-FO pruning.

## Project Overview

This project implements structured channel pruning for a SegDecNet (Segmentation + Detection) model used for surface defect detection. The goal is to reduce model size and computational complexity while maintaining high defect detection accuracy.

### Supported Pruning Methods

| Method | Description | File |
|--------|-------------|------|
| **Naive L1** | Standard L1 norm channel pruning | `run_l1_pruning.py` |
| **Naive FN** | Network Slimming (Feature Normalization gamma) | `run_fn_pruning.py` |
| **C1** | Hardware-Aligned L1 Pruning (N=32 alignment) | `run_c1_pruning.py` |
| **C2** | Defect-Aware Pruning (gradient × activation) | `run_c2_pruning.py` |
| **C3** | L1 + Activation Correlation Pruning | `run_c3_pruning.py` |
| **C4** | Taylor-FO (First-Order) Pruning | `run_c4_pruning.py` |

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (for GPU support)
- NVIDIA TensorRT 8.0+ (for inference optimization)

### Installation

1. Clone the repository and install dependencies:
```bash
pip install torch torchvision numpy matplotlib pillow tensorrt
```

2. Prepare the KSDD2 dataset:
```bash
# Place your KSDD2 dataset in ./datasets/KSDD2/
```

### Run the Full Pipeline

Execute the complete pruning and evaluation pipeline:

```bash
bash run.sh
```

This script will:
1. Profile model architectures
2. Export models to ONNX format
3. Compile TensorRT FP16 engines
4. Evaluate all models
5. Generate benchmark charts

### Run Individual Steps

```bash
# Profile models (calculate MACs and parameters)
python profile_models.py

# Export to ONNX
python export_onnx.py --model baseline_best.pth --output baseline.onnx

# Compile TensorRT engine
python compile_tensorrt.py --model baseline.onnx --mode fp16

# Evaluate models
python evaluate_models.py

# Generate charts
python generate_charts.py
```

## Project Structure

```
.
├── models.py                 # SegDecNet model definition
├── dataset_ksdd2.py          # KSDD2 dataset loader
├── train_net.py              # Baseline model training
├── run_l1_pruning.py         # Naive L1 pruning
├── run_fn_pruning.py         # Naive FN (Network Slimming) pruning
├── run_c1_pruning.py         # C1: Hardware-aligned L1 pruning
├── run_c2_pruning.py         # C2: Defect-aware pruning
├── run_c3_pruning.py         # C3: L1 + Activation correlation
├── run_c4_pruning.py         # C4: Taylor-FO pruning
├── profile_models.py         # Model profiling (MACs, params)
├── export_onnx.py            # Export to ONNX format
├── compile_tensorrt.py       # TensorRT engine compilation
├── prepare_calibration_dataset.py  # INT8 calibration data
├── evaluate_models.py        # Model evaluation
├── generate_charts.py        # Generate result charts
├── run.sh                    # End-to-end pipeline script
├── README.md                 # This file
└── README_CN.md              # Chinese documentation
```

## Pruning Methods Detail

### Naive L1 (L1 Norm Pruning)
Removes channels based on the L1 norm of their weights. Channels with smaller L1 norms are considered less important and are pruned first.

### Naive FN (Network Slimming)
Prunes channels based on the scale parameters (gamma) of the Feature Normalization layers. This method leverages batch normalization to identify and remove unimportant channels.

### C1: Hardware-Aligned L1 Pruning
Extends L1 pruning with hardware alignment constraints (N=32). This ensures the pruned model has channel counts that are multiples of 32, which is beneficial for GPU memory access patterns and compute efficiency.

### C2: Defect-Aware Pruning
Uses gradient × activation to identify channels important for defect detection. Channels that have high gradients with respect to the classification loss and high activation values are preserved.

### C3: L1 + Activation Correlation Pruning
An incremental innovation over L1 pruning that penalizes redundant channels based on the Pearson correlation of their activation patterns. Channels with high correlation are considered redundant and receive a penalty in the importance score.

### C4: Taylor-FO Pruning
Uses Taylor First-Order importance scores calculated as the sum of absolute values of (weight × gradient). This method provides a more accurate estimate of the impact of pruning each channel.

## Output Files

After running the pipeline, the following files are generated:

- **Pruned Models**: `*.pth` files (e.g., `naive_l1_r20.pth`, `ours_c3_r40.pth`)
- **ONNX Models**: `*.onnx` files for cross-platform deployment
- **TensorRT Engines**: `*_fp16.engine` files for optimized inference
- **Evaluation Results**: `real_evaluation_results.json`
- **Charts**: `chart1_master_benchmark.png`

## Model Naming Convention

Format: `{method}_r{rate}`

- `naive_l1_r20`: Naive L1 pruning with 20% pruning rate
- `ours_c3_r40`: C3 pruning with 40% pruning rate
- `baseline_best`: Baseline (unpruned) model

## Acknowledgments

This project is based on the official PyTorch implementation of:

**Mixed supervision for surface-defect detection: from weakly to fully supervised learning**
- Published in: Computers in Industry 2021
- GitHub: https://github.com/vicoslab/mixed-segdec-net-comind2021

```bibtex
@article{Bozic2021COMIND,
  author = {Bo{\v{z}}i{\v{c}}, Jakob and Tabernik, Domen and 
  Sko{\v{c}}aj, Danijel},
  journal = {Computers in Industry},
  title = {{Mixed supervision for surface-defect detection: from weakly to fully supervised learning}},
  year = {2021}
}
```

