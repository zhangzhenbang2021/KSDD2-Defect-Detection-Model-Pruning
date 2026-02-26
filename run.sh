# !/bin/bash

# =============================================================================
# KSDD2 Defect Detection Model Pruning - End-to-End Pipeline Script
# =============================================================================

set -e  

echo "======================================================================"
echo "KSDD2 Defect Detection - Model Pruning Pipeline"
echo "======================================================================"

# -----------------------------------------------------------------------------
# Step 0: Environment Check
# -----------------------------------------------------------------------------
echo ""
echo "[Step 0] Checking environment..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# -----------------------------------------------------------------------------
# Step 1: Baseline Model Training
# -----------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "[Step 1] Baseline Model Training (50 epochs)"
echo "======================================================================"

python -u train_net.py \
    --GPU=0 \
    --RUN_NAME=N_246 \
    --DATASET=KSDD2 \
    --DATASET_PATH=./datasets/KSDD2/ \
    --EPOCHS=50 \
    --LEARNING_RATE=0.01 \
    --DELTA_CLS_LOSS=1 \
    --BATCH_SIZE=10 \
    --WEIGHTED_SEG_LOSS=True \
    --WEIGHTED_SEG_LOSS_P=2 \
    --WEIGHTED_SEG_LOSS_MAX=3 \
    --DYN_BALANCED_LOSS=True \
    --GRADIENT_ADJUSTMENT=True \
    --NUM_SEGMENTED=246 \
    --RESULTS_PATH=./results \
    --VALIDATE=True \
    --VALIDATION_N_EPOCHS=5 \
    --USE_BEST_MODEL=False \
    --SAVE_IMAGES=False


# -----------------------------------------------------------------------------
# Step 2: Naive A - L1 Norm Channel Pruning
# -----------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "[Step 2] Naive A - L1 Norm Channel Pruning"
echo "======================================================================"

echo "--- L1 Pruning 20% 40% 60% ---"
python run_l1_pruning.py 

# -----------------------------------------------------------------------------
# Step 3: Naive B - Network Slimming (FN-gamma) Channel Pruning
# -----------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "[Step 3] Naive B - Network Slimming (FN-gamma) Channel Pruning"
echo "======================================================================"

echo "--- FN Pruning 20% 40% 60% ---"
python run_fn_pruning.py 


# -----------------------------------------------------------------------------
# Step 4: C1 - Hardware-Aligned L1 Structured Pruning
# -----------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "[Step 4] C1 - Hardware-Aligned L1 Structured Pruning (N=32)"
echo "======================================================================"

echo "--- C1 Pruning 20% 40% 60% ---"
python run_c1_pruning.py 


# -----------------------------------------------------------------------------
# Step 5: C2 - Pure Defect-Aware Pruning
# -----------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "[Step 5] C2 - Pure Defect-Aware Pruning"
echo "======================================================================"

echo "--- C2 Pruning 20% 40% 60% ---"
python run_c2_pruning.py 


# -----------------------------------------------------------------------------
# Step 6: C3 - L1 + Activation Correlation Structured Pruning
# -----------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "[Step 6] C3 - L1 + Activation Correlation Pruning"
echo "======================================================================"

echo "--- C3 Pruning 20% 40% 60% ---"
python run_c3_pruning.py 


# -----------------------------------------------------------------------------
# Step 7: C4 - Taylor-FO (Gradient-Weighted L1) Structured Pruning
# -----------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "[Step 7] C4 - Taylor-FO Gradient-Weighted L1 Structured Pruning"
echo "======================================================================"

echo "--- C4 Pruning 20% 40% 60% ---"
python run_c4_pruning.py 


# -----------------------------------------------------------------------------
# Step 8: Architecture Acceptance & Static Data Profiling
# -----------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "[Step 8] Architecture Acceptance & Static Data Profiling"
echo "======================================================================"
python profile_models.py

# -----------------------------------------------------------------------------
# Step 9: ONNX Export
# -----------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "[Step 9] ONNX Export"
echo "======================================================================"

MODELS=(
    "baseline_best.pth:baseline.onnx"
    "naive_l1_r20.pth:naive_l1_r20.onnx"
    "naive_l1_r40.pth:naive_l1_r40.onnx"
    "naive_l1_r60.pth:naive_l1_r60.onnx"
    "naive_fn_r20.pth:naive_fn_r20.onnx"
    "naive_fn_r40.pth:naive_fn_r40.onnx"
    "naive_fn_r60.pth:naive_fn_r60.onnx"
    "ours_c1_r20.pth:ours_c1_r20.onnx"
    "ours_c1_r40.pth:ours_c1_r40.onnx"
    "ours_c1_r60.pth:ours_c1_r60.onnx"
    "ours_c2_r20.pth:ours_c2_r20.onnx"
    "ours_c2_r40.pth:ours_c2_r40.onnx"
    "ours_c2_r60.pth:ours_c2_r60.onnx"
    "ours_c3_r20.pth:ours_c3_r20.onnx"
    "ours_c3_r40.pth:ours_c3_r40.onnx"
    "ours_c3_r60.pth:ours_c3_r60.onnx"
    "ours_c4_r20.pth:ours_c4_r20.onnx"
    "ours_c4_r40.pth:ours_c4_r40.onnx"
    "ours_c4_r60.pth:ours_c4_r60.onnx"
)

for model_pair in "${MODELS[@]}"; do
    pth_file="${model_pair%%:*}"
    onnx_file="${model_pair##*:}"
    if [ -f "$pth_file" ] && [ ! -f "$onnx_file" ]; then
        echo "Exporting: $pth_file -> $onnx_file"
        python export_onnx.py --model $pth_file --output $onnx_file
    else
        echo "Skipping: $onnx_file (already exists)"
    fi
done

# -----------------------------------------------------------------------------
# Step 10: Prepare INT8 Calibration Dataset
# -----------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "[Step 10] Prepare INT8 Calibration Dataset"
echo "======================================================================"
python prepare_calibration_dataset.py

# -----------------------------------------------------------------------------
# Step 11: TensorRT Inference Engine Compilation
# -----------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "[Step 11] TensorRT Inference Engine Compilation (FP16)"
echo "======================================================================"

# Compile FP16 engines
ONNX_FILES=(
    "baseline.onnx"
    "naive_l1_r60.onnx"
    "naive_fn_r60.onnx"
    "ours_c1_r60.onnx"
    "ours_c2_r60.onnx"
    "ours_c3_r60.onnx"
    "ours_c4_r60.onnx"
)

for onnx_file in "${ONNX_FILES[@]}"; do
    engine_file="${onnx_file%.onnx}_fp16.engine"
    if [ -f "$onnx_file" ] && [ ! -f "$engine_file" ]; then
        echo "Compiling FP16: $onnx_file -> $engine_file"
        python compile_tensorrt.py --model $onnx_file --mode fp16
    else
        echo "Skipping: $engine_file (already exists)"
    fi
done

# -----------------------------------------------------------------------------
# Step 12: Model Evaluation & Result Chart Generation
# -----------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "[Step 12] Model Evaluation & Result Chart Generation"
echo "======================================================================"
python evaluate_models.py
python generate_charts.py

# -----------------------------------------------------------------------------
# Complete
# -----------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo "Pipeline Complete!"
echo "======================================================================"
echo ""
echo "Generated files:"
echo "  - Baseline model: baseline_best.pth"
echo "  - Pruned models: naive_l1_*, naive_fn_*, ours_c1_*, ours_c2_*, ours_c3_*, ours_c4_*"
echo "  - ONNX models: *.onnx"
echo "  - TensorRT engines: *_fp16.engine"
echo "  - Evaluation results: real_evaluation_results.json"
echo "  - Charts: chart*.png"
echo ""

