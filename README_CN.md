# KSDD2 缺陷检测模型剪枝

一个端到端的通道剪枝流程，用于 KSDD2（Kaggle 表面缺陷检测）缺陷检测模型，支持多种剪枝策略，包括 L1 范数、Network Slimming、硬件对齐剪枝、缺陷感知剪枝、激活相关剪枝和 Taylor-FO 剪枝。

## 项目概述

本项目实现了用于表面缺陷检测的 SegDecNet（分割+检测）模型的结构化通道剪枝。目标是降低模型大小和计算复杂度，同时保持高缺陷检测精度。

### 支持的剪枝方法

| 方法 | 描述 | 文件 |
|------|------|------|
| **Naive L1** | 标准 L1 范数通道剪枝 | `run_l1_pruning.py` |
| **Naive FN** | Network Slimming（特征归一化 gamma） | `run_fn_pruning.py` |
| **C1** | 硬件对齐 L1 剪枝（N=32 对齐） | `run_c1_pruning.py` |
| **C2** | 缺陷感知剪枝（梯度 × 激活） | `run_c2_pruning.py` |
| **C3** | L1 + 激活相关剪枝 | `run_c3_pruning.py` |
| **C4** | Taylor-FO（一阶）剪枝 | `run_c4_pruning.py` |

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+（需要 GPU 支持）
- NVIDIA TensorRT 8.0+（用于推理优化）

### 安装

1. 克隆仓库并安装依赖：
```bash
pip install torch torchvision numpy matplotlib pillow tensorrt
```

2. 准备 KSDD2 数据集：
```bash
# 将 KSDD2 数据集放置在 ./datasets/KSDD2/ 目录下
```

### 运行完整流程

执行完整的剪枝和评估流程：

```bash
bash run.sh
```

该脚本将：
1. 分析模型架构
2. 导出 ONNX 模型
3. 编译 TensorRT FP16 引擎
4. 评估所有模型
5. 生成基准图表

### 运行单个步骤

```bash
# 分析模型（计算 MACs 和参数量）
python profile_models.py

# 导出为 ONNX
python export_onnx.py --model baseline_best.pth --output baseline.onnx

# 编译 TensorRT 引擎
python compile_tensorrt.py --model baseline.onnx --mode fp16

# 评估模型
python evaluate_models.py

# 生成图表
python generate_charts.py
```

## 项目结构

```
.
├── models.py                 # SegDecNet 模型定义
├── dataset_ksdd2.py          # KSDD2 数据集加载器
├── train_net.py              # 基线模型训练
├── run_l1_pruning.py         # Naive L1 剪枝
├── run_fn_pruning.py         # Naive FN (Network Slimming) 剪枝
├── run_c1_pruning.py         # C1: 硬件对齐 L1 剪枝
├── run_c2_pruning.py         # C2: 缺陷感知剪枝
├── run_c3_pruning.py         # C3: L1 + 激活相关剪枝
├── run_c4_pruning.py         # C4: Taylor-FO 剪枝
├── profile_models.py         # 模型分析（MACs、参数）
├── export_onnx.py            # 导出为 ONNX 格式
├── compile_tensorrt.py       # TensorRT 引擎编译
├── prepare_calibration_dataset.py  # INT8 校准数据
├── evaluate_models.py        # 模型评估
├── generate_charts.py        # 生成结果图表
├── run.sh                    # 端到端流程脚本
├── README.md                 # 英文文档
└── README_CN.md              # 本文件
```

## 剪枝方法详解

### Naive L1（L1 范数剪枝）
根据权重的 L1 范数移除通道。L1 范数较小的通道被认为不重要，优先被剪枝。

### Naive FN（Network Slimming）
根据特征归一化层（Feature Normalization）的缩放参数（gamma）来剪枝通道。该方法利用批量归一化来识别和移除不重要的通道。

### C1：硬件对齐 L1 剪枝
在 L1 剪枝基础上增加硬件对齐约束（N=32）。这确保剪枝后的模型通道数是 32 的倍数，有利于 GPU 内存访问模式和计算效率。

### C2：缺陷感知剪枝
使用梯度 × 激活来识别对缺陷检测重要的通道。具有高分类损失梯度和高激活值的通道会被保留。

### C3：L1 + 激活相关剪枝
在 L1 剪枝基础上的增量创新，根据激活模式的皮尔逊相关性对冗余通道进行惩罚。高相关性的通道被认为冗余，在重要性评分中会受到惩罚。

### C4：Taylor-FO 剪枝
使用 Taylor 一阶重要性评分，计算方式为 |weight × gradient| 的绝对值之和。该方法提供了更准确的剪枝影响估计。

## 输出文件

运行流程后，将生成以下文件：

- **剪枝模型**：`*.pth` 文件（如 `naive_l1_r20.pth`、`ours_c3_r40.pth`）
- **ONNX 模型**：`*.onnx` 文件，用于跨平台部署
- **TensorRT 引擎**：`*_fp16.engine` 文件，用于优化推理
- **评估结果**：`real_evaluation_results.json`
- **图表**：`chart1_master_benchmark.png`

## 模型命名规范

格式：`{方法}_r{剪枝率}`

- `naive_l1_r20`：20% 剪枝率的 Naive L1 剪枝
- `ours_c3_r40`：40% 剪枝率的 C3 剪枝
- `baseline_best`：基线（未剪枝）模型

## 致谢

本项目基于以下官方 PyTorch 实现：

**Mixed supervision for surface-defect detection: from weakly to fully supervised learning**
- 发表期刊：Computers in Industry 2021
- GitHub：https://github.com/vicoslab/mixed-segdec-net-comind2021

```bibtex
@article{Bozic2021COMIND,
  author = {Bo{\v{z}}i{\v{c}}, Jakob and Tabernik, Domen and 
  Sko{\v{c}}aj, Danijel},
  journal = {Computers in Industry},
  title = {{Mixed supervision for surface-defect detection: from weakly to fully supervised learning}},
  year = {2021}
}
```

