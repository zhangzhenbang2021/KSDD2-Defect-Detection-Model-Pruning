"""
Generate real Grad-CAM visualizations using actual KSDD2 dataset with REAL defects
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import FeatureNorm, SegDecNet


class ConvBlock(nn.Module):
    """Configurable Conv Block for pruned models"""
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.fn = FeatureNorm(num_features=out_channels, eps=0.001)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.fn(x)
        x = self.relu(x)
        return x


class GradientMultiplyLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask_bw):
        ctx.save_for_backward(mask_bw)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        mask_bw, = ctx.saved_tensors
        return grad_output.mul(mask_bw), None


class PrunedSegDecNet(nn.Module):
    """SegDecNet with configurable channel dimensions"""
    def __init__(self, device, input_width, input_height, input_channels, channel_config):
        super().__init__()
        
        c1 = channel_config.get('c1', 32)
        c2 = channel_config.get('c2', 64)
        c3 = channel_config.get('c3', 64)
        c4 = channel_config.get('c4', 64)
        c5 = channel_config.get('c5', 1024)
        
        self.v1 = ConvBlock(input_channels, c1, 5, 2)
        self.pool1 = nn.MaxPool2d(2)
        self.v2 = ConvBlock(c1, c2, 5, 2)
        self.v3 = ConvBlock(c2, c3, 5, 2)
        self.v4 = ConvBlock(c3, c4, 5, 2)
        self.pool2 = nn.MaxPool2d(2)
        self.v5 = ConvBlock(c4, c4, 5, 2)
        self.v6 = ConvBlock(c4, c4, 5, 2)
        self.v7 = ConvBlock(c4, c4, 5, 2)
        self.v8 = ConvBlock(c4, c4, 5, 2)
        self.pool3 = nn.MaxPool2d(2)
        self.v9 = ConvBlock(c4, c5, 15, 7)
        
        self.seg_mask = nn.Sequential(
            nn.Conv2d(c5, 1, 1, padding=0, bias=False),
            FeatureNorm(num_features=1, eps=0.001, include_bias=False))
        
        self.extractor = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(c5 + 1, 8, 5, 2),
            nn.MaxPool2d(2),
            ConvBlock(8, 16, 5, 2),
            nn.MaxPool2d(2),
            ConvBlock(16, 32, 5, 2))
        
        self.global_max_pool_feat = nn.MaxPool2d(kernel_size=32)
        self.global_avg_pool_feat = nn.AvgPool2d(kernel_size=32)
        self.global_max_pool_seg = nn.MaxPool2d(kernel_size=(input_height // 8, input_width // 8))
        self.global_avg_pool_seg = nn.AvgPool2d(kernel_size=(input_height // 8, input_width // 8))
        
        self.fc = nn.Linear(in_features=66, out_features=1)
        
        self.volume_lr_multiplier_layer = GradientMultiplyLayer.apply
        self.glob_max_lr_multiplier_layer = GradientMultiplyLayer.apply
        self.glob_avg_lr_multiplier_layer = GradientMultiplyLayer.apply
        
        self.device = device
        
    def set_gradient_multipliers(self, multiplier):
        self.volume_lr_multiplier_mask = (torch.ones((1,)) * multiplier).to(self.device)
        self.glob_max_lr_multiplier_mask = (torch.ones((1,)) * multiplier).to(self.device)
        self.glob_avg_lr_multiplier_mask = (torch.ones((1,)) * multiplier).to(self.device)
        
    def forward(self, input):
        x = self.v1(input)
        x = self.pool1(x)
        x = self.v2(x)
        x = self.v3(x)
        x = self.v4(x)
        x = self.pool2(x)
        x = self.v5(x)
        x = self.v6(x)
        x = self.v7(x)
        x = self.v8(x)
        x = self.pool3(x)
        volume = self.v9(x)
        
        seg_mask = self.seg_mask(volume)
        cat = torch.cat([volume, seg_mask], dim=1)
        cat = self.volume_lr_multiplier_layer(cat, self.volume_lr_multiplier_mask)
        
        features = self.extractor(cat)
        global_max_feat = torch.max(torch.max(features, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        global_avg_feat = torch.mean(features, dim=(-1, -2), keepdim=True)
        global_max_seg = torch.max(torch.max(seg_mask, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        global_avg_seg = torch.mean(seg_mask, dim=(-1, -2), keepdim=True)
        
        global_max_feat = global_max_feat.reshape(global_max_feat.size(0), -1)
        global_avg_feat = global_avg_feat.reshape(global_avg_feat.size(0), -1)
        global_max_seg = global_max_seg.reshape(global_max_seg.size(0), -1)
        global_max_seg = self.glob_max_lr_multiplier_layer(global_max_seg, self.glob_max_lr_multiplier_mask)
        global_avg_seg = global_avg_seg.reshape(global_avg_seg.size(0), -1)
        global_avg_seg = self.glob_avg_lr_multiplier_layer(global_avg_seg, self.glob_avg_lr_multiplier_mask)
        
        fc_in = torch.cat([global_max_feat, global_avg_feat, global_max_seg, global_avg_seg], dim=1)
        fc_in = fc_in.reshape(fc_in.size(0), -1)
        prediction = self.fc(fc_in)
        return prediction, seg_mask


def find_defect_samples(root_dir, split='train', limit=50):
    """Find samples with actual defects"""
    import torchvision.transforms as transforms
    
    defect_samples = []
    
    files = [f.replace('_GT.png', '') for f in os.listdir(os.path.join(root_dir, split)) 
             if f.endswith('_GT.png')]
    
    for img_id in files:
        mask_path = os.path.join(root_dir, split, f'{img_id}_GT.png')
        mask = np.array(Image.open(mask_path).convert('L'))
        defect_pixels = np.sum(mask > 0)
        
        if defect_pixels > 0:
            # Load image
            img_path = os.path.join(root_dir, split, f'{img_id}.png')
            img = Image.open(img_path).convert('RGB')
            
            # Resize
            img = img.resize((232, 640), Image.BILINEAR)
            mask = Image.fromarray(mask).resize((232, 640), Image.NEAREST)
            
            # Convert to tensor
            img_tensor = transforms.ToTensor()(img)
            img_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)
            mask_tensor = transforms.ToTensor()(mask)
            
            defect_samples.append({
                'img_id': img_id,
                'img_tensor': img_tensor,
                'mask_tensor': mask_tensor,
                'img_np': np.array(img),
                'mask_np': np.array(mask),
                'defect_pixels': defect_pixels
            })
    
    # Sort by defect size
    defect_samples.sort(key=lambda x: x['defect_pixels'])
    
    return defect_samples[:limit]


def get_channel_config(state_dict):
    """Extract channel config from state dict"""
    config = {'c1': 32, 'c2': 64, 'c3': 64, 'c4': 64, 'c5': 1024}
    for k in state_dict.keys():
        if 'v1.conv.weight' in k:
            config['c1'] = state_dict[k].shape[0]
        elif 'v2.conv.weight' in k:
            config['c2'] = state_dict[k].shape[0]
        elif 'v3.conv.weight' in k:
            config['c3'] = state_dict[k].shape[0]
        elif 'v5.conv.weight' in k:
            config['c4'] = state_dict[k].shape[0]
        elif 'v9.conv.weight' in k:
            config['c5'] = state_dict[k].shape[0]
    return config


def load_model(model_path, device):
    """Load model from checkpoint"""
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    config = get_channel_config(state_dict)
    is_pruned = config != {'c1': 32, 'c2': 64, 'c3': 64, 'c4': 64, 'c5': 1024}
    
    if is_pruned:
        model = PrunedSegDecNet(device, 232, 640, 3, config)
    else:
        model = SegDecNet(device, 232, 640, 3)
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    model.set_gradient_multipliers(1.0)
    
    return model


def compute_gradcam(model, input_tensor, target_layer, device):
    """Compute Grad-CAM for a specific layer"""
    model.eval()
    
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
    
    def forward_hook(module, input, output):
        activations.append(output.detach())
    
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)
    
    input_tensor = input_tensor.to(device).requires_grad_(True)
    decision, seg_mask = model(input_tensor)
    
    model.zero_grad()
    decision.squeeze().backward(retain_graph=True)
    
    if len(gradients) > 0 and len(activations) > 0:
        grad = gradients[0]
        act = activations[0]
        
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), 
                           mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
    else:
        cam = None
    
    handle_forward.remove()
    handle_backward.remove()
    
    return cam


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find REAL defect samples
    print("\nFinding REAL defect samples from KSDD2 train set...")
    defect_samples = find_defect_samples('./datasets/KSDD2', split='train', limit=50)
    print(f"Found {len(defect_samples)} defect samples")
    
    if len(defect_samples) == 0:
        print("ERROR: No defect samples found!")
        return
    
    # Select 3 diverse samples: small, medium, large
    indices = [0, len(defect_samples)//2, len(defect_samples)-1]
    selected_samples = [defect_samples[i] for i in indices]
    
    print(f"\nSelected samples:")
    for i, s in enumerate(selected_samples):
        print(f"  {i+1}. ID: {s['img_id']}, Defect pixels: {s['defect_pixels']}")
    
    # Load models
    print("\nLoading models...")
    models = {}
    model_paths = {
        'naive_l1': 'naive_l1_r60.pth',
        'ours_c2': 'ours_c2_r60.pth'
    }
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            print(f"  Loading {path}...")
            models[name] = load_model(path, device)
        else:
            print(f"  Warning: {path} not found, skipping")
    
    if len(models) == 0:
        print("ERROR: No models found!")
        return
    
    # Generate Grad-CAM
    print("\nGenerating Grad-CAM visualizations...")
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    columns = ['Input Image', 'GT Mask', 'Naive L1 (60%)', 'Ours C2 (60%)']
    for j, col in enumerate(columns):
        axes[0, j].set_title(col, fontsize=12, fontweight='bold')
    
    sample_sizes = ['Small Defect', 'Medium Defect', 'Large Defect']
    
    for row_idx, sample in enumerate(selected_samples):
        # Input image
        axes[row_idx, 0].imshow(sample['img_np'])
        axes[row_idx, 0].set_ylabel(sample_sizes[row_idx], fontsize=10, rotation=0, ha='right', va='center')
        axes[row_idx, 0].axis('off')
        
        # GT Mask
        axes[row_idx, 1].imshow(sample['mask_np'], cmap='Reds')
        axes[row_idx, 1].axis('off')
        
        # Generate Grad-CAM for each model
        for model_idx, (model_name, model) in enumerate(models.items()):
            # Get target layer (last conv before seg)
            target_layer = None
            for name, module in model.named_modules():
                if 'v9.conv' in name:
                    target_layer = module
                    break
            
            if target_layer is None:
                for name, module in model.named_modules():
                    if isinstance(module, nn.Conv2d):
                        target_layer = module
                        break
            
            if target_layer is not None:
                cam = compute_gradcam(model, sample['img_tensor'].unsqueeze(0), target_layer, device)
                
                if cam is not None:
                    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                    
                    axes[row_idx, 2 + model_idx].imshow(sample['img_np'])
                    axes[row_idx, 2 + model_idx].imshow(cam, cmap='jet', alpha=0.5, vmin=0, vmax=1)
                    axes[row_idx, 2 + model_idx].axis('off')
                else:
                    axes[row_idx, 2 + model_idx].text(0.5, 0.5, 'N/A', ha='center', va='center')
                    axes[row_idx, 2 + model_idx].axis('off')
            else:
                axes[row_idx, 2 + model_idx].text(0.5, 0.5, 'No Layer', ha='center', va='center')
                axes[row_idx, 2 + model_idx].axis('off')
    
    # Handle missing models
    if len(models) < 2:
        for row_idx in range(3):
            for col_idx in range(2):
                if col_idx + 2 < 4:
                    axes[row_idx, col_idx + 2].axis('off')
    
    plt.suptitle('Chart 4: Real Grad-CAM on KSDD2 Defect Samples (60% Pruning)\n'
                 'Using REAL defect images and masks from KSDD2 dataset', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('chart4_gradcam.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print("\n✓ chart4_gradcam.png saved successfully!")
    print(f"  - Uses REAL KSDD2 defect images")
    print(f"  - Uses REAL Ground Truth masks with actual defects")
    print(f"  - Samples: {', '.join([s['img_id'] for s in selected_samples])}")


if __name__ == '__main__':
    main()
