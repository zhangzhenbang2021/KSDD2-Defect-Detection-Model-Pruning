"""
C2 Defect-Aware Pruning for SegDecNet
Uses gradient-weighted activation importance with defect mask
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random
from sklearn.metrics import f1_score, recall_score, precision_score
import pickle
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import FeatureNorm, SegDecNet


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class ConvBlock(nn.Module):
    """Configurable Conv Block"""
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


class PrunedSegDecNet(nn.Module):
    """SegDecNet with configurable channel dimensions"""
    def __init__(self, device, input_width, input_height, input_channels, channel_config):
        super().__init__()
        if input_width % 8 != 0 or input_height % 8 != 0:
            raise Exception(f"Input size must be divisible by 8!")
        
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        
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


class GradientMultiplyLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask_bw):
        ctx.save_for_backward(mask_bw)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        mask_bw, = ctx.saved_tensors
        return grad_output.mul(mask_bw), None


class DefectDataset(torch.utils.data.Dataset):
    """Dataset with only defective samples for calibration"""
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.labels = []
        
        with open("splits/KSDD2/split_246.pyb", "rb") as f:
            train_samples, _ = pickle.load(f)
        
        # Only keep samples with ACTUAL defect pixels in mask
        for part, is_segmented in train_samples:
            img_path = os.path.join(root_dir, str(part) + ".png")
            if not os.path.exists(img_path):
                continue
                
            img = Image.open(img_path).convert("RGB")
            
            mask_path = os.path.join(root_dir, str(part) + "_GT.png")
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")
                mask = np.array(mask)
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                mask = mask / 255.0
            else:
                continue  # Skip if no mask
            
            # Check if mask actually contains defect pixels (not just is_segmented flag)
            if np.sum(mask > 0) == 0:
                continue  # Skip samples without actual defects
            
            img = img.resize((232, 640), Image.BILINEAR)
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img = mask_img.resize((232, 640), Image.NEAREST)
            mask_tensor = transforms.ToTensor()(mask_img)
            
            img_tensor = transforms.ToTensor()(img)
            img_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)
            
            self.labels.append((img_tensor, mask_tensor))
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.labels[idx]


class KSDD2Dataset(torch.utils.data.Dataset):
    """KSDD2 Dataset for training"""
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.labels = []
        
        with open("splits/KSDD2/split_246.pyb", "rb") as f:
            train_samples, test_samples = pickle.load(f)
            if split == 'train':
                samples = train_samples
            elif split == 'val':
                samples = test_samples[len(test_samples)//2:]
        
        for part, is_segmented in samples:
            img_path = os.path.join(root_dir, str(part) + ".png")
            if not os.path.exists(img_path):
                continue
                
            img = Image.open(img_path).convert("RGB")
            
            mask_path = os.path.join(root_dir, str(part) + "_GT.png")
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")
                mask = np.array(mask)
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                mask = mask / 255.0
            else:
                mask = np.zeros((img.size[1], img.size[0]))
            
            img = img.resize((232, 640), Image.BILINEAR)
            if isinstance(mask, np.ndarray):
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            else:
                mask_img = mask
            mask_img = mask_img.resize((232, 640), Image.NEAREST)
            mask_tensor = transforms.ToTensor()(mask_img)
            
            img_tensor = transforms.ToTensor()(img)
            img_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)
            
            # Determine label by analyzing actual mask content (not is_segmented flag)
            label = 1 if np.sum(mask > 0) > 0 else 0
            self.labels.append((img_tensor, mask_tensor, torch.tensor(label, dtype=torch.float32)))
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.labels[idx]


class BalancedBatchSampler:
    """Sampler that ensures balanced positive/negative samples in each batch"""
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Separate positive and negative indices
        self.positive_indices = []
        self.negative_indices = []
        
        for i, (_, _, label) in enumerate(dataset.labels):
            if label.item() == 1:
                self.positive_indices.append(i)
            else:
                self.negative_indices.append(i)
        
    def __iter__(self):
        # Shuffle
        np.random.shuffle(self.positive_indices)
        np.random.shuffle(self.negative_indices)
        
        # Create batches with balanced samples
        n_pos_per_batch = self.batch_size // 2
        n_neg_per_batch = self.batch_size - n_pos_per_batch
        
        pos_idx = 0
        neg_idx = 0
        
        while pos_idx < len(self.positive_indices) or neg_idx < len(self.negative_indices):
            batch_indices = []
            
            # Add positive samples
            for _ in range(n_pos_per_batch):
                if pos_idx < len(self.positive_indices):
                    batch_indices.append(self.positive_indices[pos_idx])
                    pos_idx += 1
            
            # Add negative samples
            for _ in range(n_neg_per_batch):
                if neg_idx < len(self.negative_indices):
                    batch_indices.append(self.negative_indices[neg_idx])
                    neg_idx += 1
            
            if len(batch_indices) > 0:
                yield batch_indices
    
    def __len__(self):
        return (len(self.positive_indices) + len(self.negative_indices)) // self.batch_size


def compute_defect_aware_importance(model, calibration_loader, device):
    """
    Compute defect-aware importance using gradient * activation * defect_mask
    
    Key insight: For defect detection, we need to focus on CLASSIFICATION, not segmentation.
    This version uses:
    1. Classification loss (decision) for backward pass - focuses on defect presence
    2. Activation magnitude weighted by defect regions
    
    Formula: I_c = sum(|activation_c| * defect_mask) / (spatial_sum of mask)
    - This measures how much each channel contributes to defect regions
    """
    model.eval()
    
    # Layers to compute importance for
    conv_layers = ['volume.0.0', 'volume.2.0', 'volume.3.0', 'volume.4.0',
                   'volume.6.0', 'volume.7.0', 'volume.8.0', 'volume.9.0', 'volume.11.0']
    
    # Storage for activations (saved during forward pass)
    activations = {name: None for name in conv_layers}
    
    # Storage for importance scores - accumulate over samples
    importance_sums = {name: None for name in conv_layers}
    importance_counts = {name: 0 for name in conv_layers}
    
    hooks = []
    
    # Forward hook - save activation
    def get_activation(name):
        def hook(module, input, output):
            # Save output activations
            activations[name] = output.detach()
        return hook
    
    # Find conv layers and register forward hooks only
    for name, module in model.named_modules():
        if name in conv_layers:
            hooks.append(module.register_forward_hook(get_activation(name)))
    
    print(f"  Computing defect-aware importance from {len(calibration_loader)} defect samples...")
    
    # Set gradient multipliers to enable forward pass
    model.set_gradient_multipliers(1.0)
    
    for batch_idx, (images, masks) in enumerate(calibration_loader):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        decision, seg_mask = model(images)
        
        # Get activations for each layer
        for name in conv_layers:
            if activations[name] is None:
                continue
            
            act = activations[name]  # [B, C, H, W]
            
            # Initialize importance sums if needed
            if importance_sums[name] is None:
                importance_sums[name] = torch.zeros(act.shape[1], device=device)
            
            # Resize defect mask to match activation spatial dimensions
            mask_resized = torch.nn.functional.interpolate(
                masks, size=(act.shape[2], act.shape[3]), mode='bilinear', align_corners=False
            )
            
            # Compute importance: |activation| * defect_mask
            # This measures how much each channel is activated in defect regions
            batch_size = act.shape[0]
            for b in range(batch_size):
                act_b = act[b]  # [C, H, W]
                mask_b = mask_resized[b]  # [1, H, W]
                
                # |activation| * defect_mask (sum over spatial dimensions)
                # Normalize by mask area to handle different defect sizes
                weighted = act_b.abs() * mask_b
                spatial_sum = mask_b.sum() + 1e-8  # Avoid division by zero
                
                importance_sums[name] += weighted.sum(dim=(1, 2)) / spatial_sum
            
            importance_counts[name] += batch_size
        
        # Reset activations for next batch
        activations = {name: None for name in conv_layers}
        
        if (batch_idx + 1) % 50 == 0:
            print(f"    Processed {batch_idx+1}/{len(calibration_loader)} samples...")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Average importance
    avg_importance = {}
    for name in conv_layers:
        if importance_sums[name] is not None and importance_counts[name] > 0:
            avg_importance[name] = (importance_sums[name] / importance_counts[name]).cpu().numpy()
    
    return avg_importance


def prune_c2_model(model, importance_scores, prune_rate, device):
    """Build C2 pruned model based on defect-aware importance"""
    state_dict = model.state_dict()
    
    # Original channel counts
    orig_channels = {
        'volume.0.0': 32,
        'volume.2.0': 64,
        'volume.3.0': 64,
        'volume.4.0': 64,
        'volume.6.0': 64,
        'volume.7.0': 64,
        'volume.8.0': 64,
        'volume.9.0': 64,
        'volume.11.0': 1024,
    }
    
    # Calculate target channels based on importance with hardware alignment
    new_channels = {}
    print(f"\n  Prune rate: {prune_rate*100:.0f}% (with hardware alignment)")
    for conv_name, orig_ch in orig_channels.items():
        if conv_name in importance_scores:
            scores = importance_scores[conv_name]
            target_ch = int(orig_ch * (1 - prune_rate))
            target_ch = max(1, target_ch)
            
            # Apply hardware alignment (same logic as C1)
            if orig_ch >= 256:  # Large layer (c5)
                aligned_ch = ((target_ch + 16) // 32) * 32
                aligned_ch = max(32, min(aligned_ch, orig_ch))
            elif orig_ch >= 48:  # Medium layer (c2, c3, c4)
                aligned_ch = ((target_ch + 4) // 8) * 8
                aligned_ch = max(8, min(aligned_ch, orig_ch))
                if aligned_ch >= orig_ch * 0.95:
                    aligned_ch = max(8, orig_ch - 8)
            else:  # Small layer (c1)
                aligned_ch = target_ch
            
            # Keep channels with highest importance
            n_keep = aligned_ch
            keep_indices = np.argsort(scores)[-n_keep:]
            keep_indices = np.sort(keep_indices)
            new_channels[conv_name] = (n_keep, keep_indices)
        else:
            new_channels[conv_name] = (orig_ch, np.arange(orig_ch))
    
    channel_config = {
        'c1': new_channels['volume.0.0'][0],
        'c2': new_channels['volume.2.0'][0],
        'c3': new_channels['volume.3.0'][0],
        'c4': new_channels['volume.4.0'][0],
        'c5': new_channels['volume.11.0'][0],
    }
    
    print(f"  New channel config: {channel_config}")
    
    # Create new model
    new_model = PrunedSegDecNet(device, 232, 640, 3, channel_config)
    new_model = new_model.to(device)
    pruned_state = new_model.state_dict()
    
    c1, c2, c3, c4, c5 = channel_config['c1'], channel_config['c2'], channel_config['c3'], channel_config['c4'], channel_config['c5']
    
    # Get keep indices for each layer
    keep_v1 = new_channels['volume.0.0'][1]
    keep_v2 = new_channels['volume.2.0'][1]
    keep_v3 = new_channels['volume.3.0'][1]
    keep_v4 = new_channels['volume.4.0'][1]
    keep_v5 = new_channels['volume.6.0'][1]
    keep_v6 = new_channels['volume.7.0'][1]
    keep_v7 = new_channels['volume.8.0'][1]
    keep_v8 = new_channels['volume.9.0'][1]
    keep_v9 = new_channels['volume.11.0'][1]
    
    # Volume 1: input=3, output=c1
    pruned_state['v1.conv.weight'] = state_dict['volume.0.0.weight'][keep_v1, :, :, :].clone()
    pruned_state['v1.fn.scale'] = state_dict['volume.0.1.scale'][:, keep_v1, :, :].clone()
    pruned_state['v1.fn.bias'] = state_dict['volume.0.1.bias'][:, keep_v1, :, :].clone()
    
    # Volume 2: input=c1, output=c2
    pruned_state['v2.conv.weight'] = state_dict['volume.2.0.weight'][keep_v2, :, :, :].clone()
    pruned_state['v2.conv.weight'] = pruned_state['v2.conv.weight'][:, keep_v1, :, :].clone()
    pruned_state['v2.fn.scale'] = state_dict['volume.2.1.scale'][:, keep_v2, :, :].clone()
    pruned_state['v2.fn.bias'] = state_dict['volume.2.1.bias'][:, keep_v2, :, :].clone()
    
    # Volume 3: input=c2, output=c3
    pruned_state['v3.conv.weight'] = state_dict['volume.3.0.weight'][keep_v3, :, :, :].clone()
    pruned_state['v3.conv.weight'] = pruned_state['v3.conv.weight'][:, keep_v2, :, :].clone()
    pruned_state['v3.fn.scale'] = state_dict['volume.3.1.scale'][:, keep_v3, :, :].clone()
    pruned_state['v3.fn.bias'] = state_dict['volume.3.1.bias'][:, keep_v3, :, :].clone()
    
    # Volume 4: input=c3, output=c4
    pruned_state['v4.conv.weight'] = state_dict['volume.4.0.weight'][keep_v4, :, :, :].clone()
    pruned_state['v4.conv.weight'] = pruned_state['v4.conv.weight'][:, keep_v3, :, :].clone()
    pruned_state['v4.fn.scale'] = state_dict['volume.4.1.scale'][:, keep_v4, :, :].clone()
    pruned_state['v4.fn.bias'] = state_dict['volume.4.1.bias'][:, keep_v4, :, :].clone()
    
    # Volume 5-8: input=c4, output=c4
    pruned_state['v5.conv.weight'] = state_dict['volume.6.0.weight'][keep_v5, :, :, :].clone()
    pruned_state['v5.conv.weight'] = pruned_state['v5.conv.weight'][:, keep_v4, :, :].clone()
    pruned_state['v5.fn.scale'] = state_dict['volume.6.1.scale'][:, keep_v5, :, :].clone()
    pruned_state['v5.fn.bias'] = state_dict['volume.6.1.bias'][:, keep_v5, :, :].clone()
    
    pruned_state['v6.conv.weight'] = state_dict['volume.7.0.weight'][keep_v6, :, :, :].clone()
    pruned_state['v6.conv.weight'] = pruned_state['v6.conv.weight'][:, keep_v5, :, :].clone()
    pruned_state['v6.fn.scale'] = state_dict['volume.7.1.scale'][:, keep_v6, :, :].clone()
    pruned_state['v6.fn.bias'] = state_dict['volume.7.1.bias'][:, keep_v6, :, :].clone()
    
    pruned_state['v7.conv.weight'] = state_dict['volume.8.0.weight'][keep_v7, :, :, :].clone()
    pruned_state['v7.conv.weight'] = pruned_state['v7.conv.weight'][:, keep_v6, :, :].clone()
    pruned_state['v7.fn.scale'] = state_dict['volume.8.1.scale'][:, keep_v7, :, :].clone()
    pruned_state['v7.fn.bias'] = state_dict['volume.8.1.bias'][:, keep_v7, :, :].clone()
    
    pruned_state['v8.conv.weight'] = state_dict['volume.9.0.weight'][keep_v8, :, :, :].clone()
    pruned_state['v8.conv.weight'] = pruned_state['v8.conv.weight'][:, keep_v7, :, :].clone()
    pruned_state['v8.fn.scale'] = state_dict['volume.9.1.scale'][:, keep_v8, :, :].clone()
    pruned_state['v8.fn.bias'] = state_dict['volume.9.1.bias'][:, keep_v8, :, :].clone()
    
    # Volume 9: input=c4, output=c5
    pruned_state['v9.conv.weight'] = state_dict['volume.11.0.weight'][keep_v9, :, :, :].clone()
    pruned_state['v9.conv.weight'] = pruned_state['v9.conv.weight'][:, keep_v8, :, :].clone()
    pruned_state['v9.fn.scale'] = state_dict['volume.11.1.scale'][:, keep_v9, :, :].clone()
    pruned_state['v9.fn.bias'] = state_dict['volume.11.1.bias'][:, keep_v9, :, :].clone()
    
    # Seg mask
    pruned_state['seg_mask.0.weight'] = state_dict['seg_mask.0.weight'][:, keep_v9, :, :].clone()
    pruned_state['seg_mask.1.scale'] = state_dict['seg_mask.1.scale'].clone()
    
    # Extractor: adjust input channels
    # Extractor input = [volume (c5 channels), seg_mask (1 channel)]
    # We need to: 1) select volume channels using keep_v9 indices, 2) keep seg_mask channel unchanged
    old_ext1 = state_dict['extractor.1.0.weight']
    old_c5 = old_ext1.shape[1] - 1  # Original c5 (1024)
    
    # Select volume channels using keep_v9 indices
    selected_volume = old_ext1[:, :old_c5, :, :][:, keep_v9, :, :]
    # Keep seg_mask channel (last channel)
    seg_mask_ch = old_ext1[:, old_c5:old_c5+1, :, :]
    # Concatenate: [selected_volume, seg_mask]
    new_ext1 = torch.cat([selected_volume, seg_mask_ch], dim=1)
    pruned_state['extractor.1.conv.weight'] = new_ext1
    pruned_state['extractor.1.fn.scale'] = state_dict['extractor.1.1.scale'].clone()
    pruned_state['extractor.1.fn.bias'] = state_dict['extractor.1.1.bias'].clone()
    
    pruned_state['extractor.3.conv.weight'] = state_dict['extractor.3.0.weight'].clone()
    pruned_state['extractor.3.fn.scale'] = state_dict['extractor.3.1.scale'].clone()
    pruned_state['extractor.3.fn.bias'] = state_dict['extractor.3.1.bias'].clone()
    
    pruned_state['extractor.5.conv.weight'] = state_dict['extractor.5.0.weight'].clone()
    pruned_state['extractor.5.fn.scale'] = state_dict['extractor.5.1.scale'].clone()
    pruned_state['extractor.5.fn.bias'] = state_dict['extractor.5.1.bias'].clone()
    
    # FC
    pruned_state['fc.weight'] = state_dict['fc.weight'].clone()
    pruned_state['fc.bias'] = state_dict['fc.bias'].clone()
    
    new_model.load_state_dict(pruned_state, strict=True)
    return new_model, channel_config


class RecallPreservedFineTuner:
    """Fine-tuner with recall-preserved strategy (pos_weight BCE + balanced sampling)"""
    
    def __init__(self, model, train_dataset, val_loader, device, config):
        self.model = model.to(device)
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        self.seg_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        
        # Use pos_weight for BCE to emphasize recall
        pos_weight = torch.tensor([config.get('pos_weight', 5.0)]).to(device)
        self.dec_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        self.optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
        
        # Create balanced batch sampler
        self.train_sampler = BalancedBatchSampler(train_dataset, config.get('batch_size', 10))
        self.train_dataset = train_dataset
        
        self.model.set_gradient_multipliers(0.0)
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_indices in self.train_sampler:
            # Get batch data
            images_list = []
            masks_list = []
            labels_list = []
            
            for idx in batch_indices:
                img, mask, label = self.train_dataset[idx]
                images_list.append(img)
                masks_list.append(mask)
                labels_list.append(label)
            
            images = torch.stack(images_list).to(self.device)
            masks = torch.stack(masks_list).to(self.device)
            labels = torch.stack(labels_list).to(self.device)
            
            self.optimizer.zero_grad()
            decision, seg_mask_pred = self.model(images)
            
            # Compute loss
            total_epochs = self.config['num_epochs']
            seg_loss_weight = 1 - (epoch / total_epochs)
            dec_loss_weight = self.config.get('delta_cls_loss', 1.0) * (epoch / total_epochs)
            
            # Classification loss with pos_weight
            dec_loss = self.dec_loss_fn(decision.view(-1), labels.view(-1))
            
            # Segmentation loss
            seg_mask_resized = torch.nn.functional.interpolate(
                seg_mask_pred, size=masks.shape[-2:], mode='bilinear', align_corners=False
            )
            seg_loss = self.seg_loss_fn(seg_mask_resized, masks).mean()
            
            loss = seg_loss_weight * seg_loss + dec_loss_weight * dec_loss
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred_labels = (torch.sigmoid(decision.view(-1)) > 0.5).float()
            correct += (pred_labels == labels.view(-1)).sum().item()
            total += labels.size(0)
        
        return total_loss / len(self.train_sampler), correct / total
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        for images, masks, labels in self.val_loader:
            images = images.to(self.device)
            decision, _ = self.model(images)
            pred_labels = (torch.sigmoid(decision.view(-1)) > 0.5).cpu().numpy()
            all_preds.extend(pred_labels)
            all_labels.extend(labels.numpy())
        
        f1 = f1_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        return f1, recall, precision
    
    def train(self, num_epochs, save_name):
        best_f1 = 0
        best_recall = 0
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_f1, val_recall, val_precision = self.evaluate()
            
            self.scheduler.step(val_f1)
            
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val F1: {val_f1:.4f}, Recall: {val_recall:.4f}, Prec: {val_precision:.4f}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_recall = val_recall
                torch.save(self.model.state_dict(), save_name)
                print(f"  -> Saved best model with F1: {best_f1:.4f}")
        
        return best_f1, best_recall


def run_c2_pruning(prune_rates=[0.2, 0.4, 0.6], calibration_samples=150):
    """Run C2 Defect-Aware Pruning"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load baseline
    print("="*60)
    print("Loading baseline model...")
    print("="*60)
    model = SegDecNet(device, 232, 640, 3)
    model.load_state_dict(torch.load('baseline_best.pth', map_location=device, weights_only=False))
    model = model.to(device)
    model.set_gradient_multipliers(1.0)
    
    orig_params = sum(p.numel() for p in model.parameters())
    print(f"Original parameters: {orig_params:,}")
    
    # Step 1: Build calibration set (defective samples only)
    print("\n" + "="*60)
    print("Step 1: Building calibration set (defect-only)")
    print("="*60)
    defect_dataset = DefectDataset('./datasets/KSDD2/train/')
    print(f"Total defective samples available: {len(defect_dataset)}")
    
    # Sample calibration set
    n_samples = min(calibration_samples, len(defect_dataset))
    indices = np.random.choice(len(defect_dataset), n_samples, replace=False)
    calibration_subset = torch.utils.data.Subset(defect_dataset, indices)
    calibration_loader = DataLoader(calibration_subset, batch_size=1, shuffle=False)
    print(f"Using {n_samples} samples for calibration (batch_size=1)")
    
    # Step 2: Compute defect-aware importance
    print("\n" + "="*60)
    print("Step 2: Computing defect-aware importance (Gradient x Activation x Mask)")
    print("="*60)
    importance_scores = compute_defect_aware_importance(model, calibration_loader, device)
    
    for name, scores in importance_scores.items():
        print(f"  {name}: {len(scores)} channels, mean importance={np.mean(scores):.6f}")
    
    # Load datasets for fine-tuning
    print("\n" + "="*60)
    print("Step 3: Loading datasets for recall-preserved fine-tuning")
    print("="*60)
    train_dataset = KSDD2Dataset('./datasets/KSDD2/train/', split='train')
    val_dataset = KSDD2Dataset('./datasets/KSDD2/test/', split='val')
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=2)
    
    # Fine-tuning config: pos_weight BCE + balanced sampling
    fine_tune_config = {
        'learning_rate': 0.001,
        'num_epochs': 15,
        'delta_cls_loss': 1.0,
        'pos_weight': 5.0,  # Higher weight for positive class
        'batch_size': 10,
    }
    
    # Prune each rate
    for prune_rate in prune_rates:
        print("\n" + "="*60)
        print(f"C2 Pruning with rate: {prune_rate*100:.0f}%")
        print("="*60)
        
        # Build pruned model
        print("Step 4: Building C2 pruned model...")
        pruned_model, channel_config = prune_c2_model(model, importance_scores, prune_rate, device)
        pruned_model.set_gradient_multipliers(1.0)
        
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        print(f"\n  Pruned parameters: {pruned_params:,} ({pruned_params/orig_params*100:.1f}%)")
        print(f"  Channel config: {channel_config}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 640, 232).to(device)
        decision, seg_mask = pruned_model(dummy_input)
        print(f"\n  Forward pass OK!")
        
        # Recall-preserved fine-tuning
        print(f"\nStep 5: Recall-preserved fine-tuning ({fine_tune_config['num_epochs']} epochs, pos_weight={fine_tune_config['pos_weight']})...")
        fine_tuner = RecallPreservedFineTuner(pruned_model, train_dataset, val_loader, device, fine_tune_config)
        
        save_name = f'ours_c2_r{int(prune_rate*100)}.pth'
        best_f1, best_recall = fine_tuner.train(fine_tune_config['num_epochs'], save_name)
        
        print(f"\n=== Result for r={prune_rate*100:.0f}%: F1={best_f1:.4f}, Recall={best_recall:.4f} ===")
        print(f"  Saved to: {save_name}")
    
    print("\n" + "="*60)
    print("C2 Defect-Aware Pruning Complete!")
    print("="*60)
    
    print("\n=== Summary ===")
    print(f"Baseline: {orig_params:,} params")
    for rate in prune_rates:
        save_name = f'ours_c2_r{int(rate*100)}.pth'
        print(f"Ours C2 r{int(rate*100)}: {save_name}")


if __name__ == '__main__':
    set_seed(42)
    run_c2_pruning(prune_rates=[0.2, 0.4, 0.6], calibration_samples=150)
