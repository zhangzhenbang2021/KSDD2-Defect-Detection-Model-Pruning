"""
Export all models to ONNX format
Handles both baseline and pruned models
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

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


def get_channel_config_from_state_dict(state_dict):
    """Extract channel config from state dict"""
    config = {'c1': 32, 'c2': 64, 'c3': 64, 'c4': 64, 'c5': 1024}
    
    # Try to find v1 conv weight
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
            config['c4'] = state_dict[k].shape[1]  # input channels = c4
        elif 'seg_mask.0.weight' in k:
            # This is the output channels
            pass
    
    return config


def export_to_onnx(model_path, output_path, device='cuda'):
    """Export a model to ONNX format"""
    print(f"Exporting {model_path} -> {output_path}")
    
    # Load state dict first to get actual channel config
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get channel config from state dict
    channel_config = get_channel_config_from_state_dict(state_dict)
    print(f"  Channel config: {channel_config}")
    
    # Check if it's a pruned model (non-default config)
    is_pruned = channel_config != {'c1': 32, 'c2': 64, 'c3': 64, 'c4': 64, 'c5': 1024}
    
    if is_pruned:
        # Use PrunedSegDecNet for pruned models
        model = PrunedSegDecNet(device, 232, 640, 3, channel_config)
    else:
        # Use original SegDecNet for baseline
        model = SegDecNet(device, 232, 640, 3)
    
    # Load weights
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    model.set_gradient_multipliers(1.0)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 640, 232, device=device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['decision', 'seg_mask'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'decision': {0: 'batch_size'},
            'seg_mask': {0: 'batch_size'}
        }
    )
    
    # Verify ONNX file
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved: {output_path} ({file_size:.2f} MB)")
    
    # Try to load and verify
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"  ✓ ONNX model verified")
    except ImportError:
        print(f"  (onnx not installed, skipping verification)")
    except Exception as e:
        print(f"  ✗ ONNX verification failed: {e}")
    
    return True


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Models to export
    models = [
        ('baseline_best.pth', 'baseline.onnx'),
        ('naive_l1_r20.pth', 'naive_l1_r20.onnx'),
        ('naive_l1_r40.pth', 'naive_l1_r40.onnx'),
        ('naive_l1_r60.pth', 'naive_l1_r60.onnx'),
        ('naive_fn_r20.pth', 'naive_fn_r20.onnx'),
        ('naive_fn_r40.pth', 'naive_fn_r40.onnx'),
        ('naive_fn_r60.pth', 'naive_fn_r60.onnx'),
        ('ours_c1_r20.pth', 'ours_c1_r20.onnx'),
        ('ours_c1_r40.pth', 'ours_c1_r40.onnx'),
        ('ours_c1_r60.pth', 'ours_c1_r60.onnx'),
        ('ours_c2_r20.pth', 'ours_c2_r20.onnx'),
        ('ours_c2_r40.pth', 'ours_c2_r40.onnx'),
        ('ours_c2_r60.pth', 'ours_c2_r60.onnx'),
        ('ours_c3_r20.pth', 'ours_c3_r20.onnx'),
        ('ours_c3_r40.pth', 'ours_c3_r40.onnx'),
        ('ours_c3_r60.pth', 'ours_c3_r60.onnx'),
        ('ours_c4_r20.pth', 'ours_c4_r20.onnx'),
        ('ours_c4_r40.pth', 'ours_c4_r40.onnx'),
        ('ours_c4_r60.pth', 'ours_c4_r60.onnx'),
    ]
    
    print("\n" + "="*60)
    print("Exporting Models to ONNX")
    print("="*60)
    
    success_count = 0
    for model_path, onnx_path in models:
        if not os.path.exists(model_path):
            print(f"\nSkipping {model_path} (not found)")
            continue
        
        try:
            export_to_onnx(model_path, onnx_path, device)
            success_count += 1
        except Exception as e:
            print(f"\n✗ Failed to export {model_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"Export Complete! {success_count}/{len(models)} models exported")
    print("="*60)
    
    # List exported files
    print("\nExported files:")
    for _, onnx_path in models:
        if os.path.exists(onnx_path):
            size = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"  {onnx_path}: {size:.2f} MB")


if __name__ == '__main__':
    main()
