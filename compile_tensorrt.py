"""
TensorRT Engine Compilation Script
Works with TensorRT 10.x
"""

import os
import sys
import argparse
import numpy as np
import tensorrt as trt

# Calibration dataset import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_calibration_dataset import preprocess_image, OUTPUT_DIR


class CalibrationDataset:
    """Simple calibration dataset for TensorRT INT8"""
    
    def __init__(self, image_dir, batch_size=1):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.current_idx = 0
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = preprocess_image(img_path)
        return img.astype(np.float32)
    
    def reset(self):
        self.current_idx = 0


class TensorRTCalibrator(trt.IInt8EntropyCalibrator2):
    """TensorRT INT8 Calibrator"""
    
    def __init__(self, calibration_dataset):
        trt.IInt8EntropyCalibrator2.__init__(self)
        
        self.dataset = calibration_dataset
        self.dataset.reset()
        
    def get_batch_size(self):
        return 1
    
    def get_batch(self, names):
        if self.dataset.current_idx >= len(self.dataset):
            return None
        
        data = self.dataset[self.dataset.current_idx]
        self.dataset.current_idx += 1
        
        # Return as list of pointers
        return [int(data.ctypes.data)]
    
    def read_calibration_cache(self):
        if os.path.exists('calibration_cache.cache'):
            with open('calibration_cache.cache', 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        with open('calibration_cache.cache', 'wb') as f:
            f.write(cache)


def build_engine(onnx_path, mode='fp16', calibration_dataset=None):
    """Build TensorRT engine from ONNX"""
    
    # Create builder and network
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("Failed to parse ONNX")
            for error in parser.get_error_list():
                print(error)
            return None
    
    # Get input shape
    input_tensor = network.get_input(0)
    input_shape = input_tensor.shape
    print(f"Input shape: {input_shape}")
    
    # Create config
    config = builder.create_builder_config()
    
    # Create optimization profile
    profile = builder.create_optimization_profile()

    # Detect dynamic batch: TensorRT uses -1 for dynamic dims
    shape = tuple(input_shape)

    if shape[0] == -1:
        # Dynamic batch ONNX: choose a reasonable profile
        min_shape = (1, shape[1], shape[2], shape[3])
        opt_shape = (8, shape[1], shape[2], shape[3])
        max_shape = (32, shape[1], shape[2], shape[3])
    else:
        # Static ONNX: profile must match exactly
        min_shape = opt_shape = max_shape = shape

    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    print(f"Profile shapes min/opt/max: {min_shape} / {opt_shape} / {max_shape}")
    
    # Set precision
    if mode == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
        print("Building FP16 engine...")
    elif mode == 'int8':
        if calibration_dataset is None:
            print("ERROR: INT8 requires calibration dataset")
            return None
        config.set_flag(trt.BuilderFlag.INT8)
        # Also enable FP16 for layers that don't support INT8
        config.set_flag(trt.BuilderFlag.FP16)
        print("Building INT8 engine...")
    else:
        print(f"Unknown mode: {mode}")
        return None
    
    # Set INT8 calibrator
    if mode == 'int8':
        calibrator = TensorRTCalibrator(calibration_dataset)
        config.int8_calibrator = calibrator
    
    # Build engine
    engine_bytes = builder.build_serialized_network(network, config)
    
    if engine_bytes is None:
        print("Failed to build engine")
        return None
    
    return engine_bytes


def main():
    parser = argparse.ArgumentParser(description='Compile TensorRT engine from ONNX')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--mode', type=str, default='fp16', choices=['fp16', 'int8'], help='Precision mode')
    parser.add_argument('--output', type=str, default=None, help='Output engine path')
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output is None:
        base_name = os.path.splitext(args.model)[0]
        args.output = f"{base_name}_{args.mode}.engine"
    
    print(f"Compiling {args.model} -> {args.output}")
    print(f"Mode: {args.mode}")
    
    # Prepare calibration dataset for INT8
    calibration_dataset = None
    if args.mode == 'int8':
        print(f"Using calibration dataset from {OUTPUT_DIR}")
        calibration_dataset = CalibrationDataset(OUTPUT_DIR, batch_size=1)
        print(f"Calibration samples: {len(calibration_dataset)}")
    
    # Build engine
    engine_bytes = build_engine(args.model, args.mode, calibration_dataset)
    
    if engine_bytes is not None:
        with open(args.output, 'wb') as f:
            f.write(engine_bytes)
        print(f"✓ Engine saved to {args.output}")
        print(f"  Size: {os.path.getsize(args.output) / (1024*1024):.2f} MB")
    else:
        print("✗ Failed to build engine")


if __name__ == '__main__':
    main()
