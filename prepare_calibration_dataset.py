"""
Prepare INT8 Calibration Dataset
Extracts images from training set with sufficient defect positive samples
"""

import os
import sys
import shutil
import random
import pickle
import numpy as np
from PIL import Image

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Output directory
OUTPUT_DIR = 'calibration_images'
NUM_SAMPLES = 200  # Number of calibration samples

def prepare_calibration_dataset():
    """Prepare calibration dataset - KSDD2 all samples are defective"""
    
    print(f"Preparing calibration dataset...")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Total samples: {NUM_SAMPLES}")
    
    # Load dataset splits
    with open("splits/KSDD2/split_246.pyb", "rb") as f:
        train_samples, _ = pickle.load(f)
    
    # KSDD2 - only use samples with ACTUAL defect pixels in mask
    all_samples = []
    for part, is_segmented in train_samples:
        img_path = os.path.join('./datasets/KSDD2/train/', str(part) + ".png")
        mask_path = os.path.join('./datasets/KSDD2/train/', str(part) + "_GT.png")
        
        if not os.path.exists(img_path):
            continue
        
        # Check if mask actually contains defect pixels
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask)
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = mask / 255.0
            
            # Only keep samples with actual defects
            if np.sum(mask > 0) > 0:
                all_samples.append((part, img_path))
    
    print(f"\nAvailable defect samples: {len(all_samples)}")
    
    # Randomly sample
    sampled = random.sample(all_samples, min(NUM_SAMPLES, len(all_samples)))
    
    print(f"\nSampled for calibration: {len(sampled)}")
    
    # Copy images to output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    count = 0
    for part, img_path in sampled:
        # Copy with sequential naming
        dest_path = os.path.join(OUTPUT_DIR, f"calib_{count:04d}.png")
        shutil.copy2(img_path, dest_path)
        count += 1
    
    print(f"\n✓ Copied {count} images to {OUTPUT_DIR}")
    
    # Save metadata
    metadata = {
        'num_samples': count,
    }
    
    with open(os.path.join(OUTPUT_DIR, 'metadata.txt'), 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nMetadata saved to {OUTPUT_DIR}/metadata.txt")
    print(f"  {metadata}")
    
    return metadata


def preprocess_image(img_path):
    """Preprocess image for model input"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((232, 640), Image.BILINEAR)
    
    # Convert to tensor
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # HWC to CHW
    img_array = img_array.transpose(2, 0, 1)
    
    return img_array


def create_calibration_data():
    """Create calibration data as numpy arrays"""
    
    print("\n" + "="*60)
    print("Creating calibration data")
    print("="*60)
    
    # Get all calibration images
    calib_images = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
    
    print(f"Found {len(calib_images)} calibration images")
    
    # Preprocess all images
    calib_data = []
    for img_name in calib_images:
        img_path = os.path.join(OUTPUT_DIR, img_name)
        img_tensor = preprocess_image(img_path)
        calib_data.append(img_tensor)
    
    calib_array = np.stack(calib_data, axis=0)
    print(f"Calibration data shape: {calib_array.shape}")
    print(f"Calibration data dtype: {calib_array.dtype}")
    print(f"Calibration data range: [{calib_array.min():.2f}, {calib_array.max():.2f}]")
    
    return calib_array


def test_calibration_data():
    """Test that calibration data can be loaded correctly"""
    
    print("\n" + "="*60)
    print("Testing calibration data loader")
    print("="*60)
    
    calib_data = create_calibration_data()
    
    # Test first few samples
    test_samples = min(5, len(calib_data))
    print(f"Testing {test_samples} samples...")
    
    for i in range(test_samples):
        print(f"  Sample {i}: shape={calib_data[i].shape}, dtype={calib_data[i].dtype}, "
              f"range=[{calib_data[i].min():.2f}, {calib_data[i].max():.2f}]")
    
    print(f"\n✓ Calibration data loader test passed!")


if __name__ == '__main__':
    # Prepare calibration dataset
    metadata = prepare_calibration_dataset()
    
    # Test the data loader
    test_calibration_data()
    
    print("\n" + "="*60)
    print("Calibration Dataset Preparation Complete!")
    print("="*60)
    print(f"\nTo use with TensorRT INT8 calibration:")
    print(f"  1. Images are in: {OUTPUT_DIR}/")
    print(f"  2. Use preprocess_image() function from this script")
    print(f"  3. Total samples: {metadata['num_samples']}")
