"""
Profile all models to calculate MACs (FLOPs) and Parameters
Uses theoretical calculation based on channel configurations from actual checkpoints
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SegDecNet


def get_model_config_from_ckpt(model_path):
    """Extract channel configuration from model checkpoint"""
    if not os.path.exists(model_path):
        return None
    
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    
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
            config['c4'] = state_dict[k].shape[1]
    
    return config


def calculate_params(channel_config):
    """Calculate theoretical params based on channel config"""
    c1, c2, c3, c4, c5 = channel_config['c1'], channel_config['c2'], channel_config['c3'], channel_config['c4'], channel_config['c5']
    
    # v1: 3 -> c1, 5x5 conv + fn
    params_v1 = 3 * c1 * 5 * 5 + c1 * 2  # conv + fn (scale, bias)
    
    # v2: c1 -> c2
    params_v2 = c1 * c2 * 5 * 5 + c2 * 2
    
    # v3: c2 -> c3
    params_v3 = c2 * c3 * 5 * 5 + c3 * 2
    
    # v4: c3 -> c4
    params_v4 = c3 * c4 * 5 * 5 + c4 * 2
    
    # v5-v8: c4 -> c4 (4 layers)
    params_v5v8 = 4 * (c4 * c4 * 5 * 5 + c4 * 2)
    
    # v9: c4 -> c5
    params_v9 = c4 * c5 * 15 * 15 + c5 * 2
    
    # seg_mask: c5 -> 1
    params_seg = c5 * 1 * 1 * 1 + 1 * 2
    
    # extractor
    params_ext1 = (c5 + 1) * 8 * 5 * 5 + 8 * 2
    params_ext2 = 8 * 16 * 5 * 5 + 16 * 2
    params_ext3 = 16 * 32 * 5 * 5 + 32 * 2
    
    # fc: 66 -> 1
    params_fc = 66 * 1 + 1
    
    total_params = params_v1 + params_v2 + params_v3 + params_v4 + params_v5v8 + params_v9 + params_seg + params_ext1 + params_ext2 + params_ext3 + params_fc
    
    return total_params / 1e6  # M params


def calculate_macs(channel_config):
    """Calculate theoretical FLOPs based on channel config"""
    c1, c2, c3, c4, c5 = channel_config['c1'], channel_config['c2'], channel_config['c3'], channel_config['c4'], channel_config['c5']
    
    h, w = 640, 232
    
    # v1: 3 -> c1, 5x5 conv
    macs_v1 = 2 * c1 * 3 * 5 * 5 * (h//2) * (w//2)
    
    # v2: c1 -> c2
    macs_v2 = 2 * c2 * c1 * 5 * 5 * (h//4) * (w//4)
    
    # v3: c2 -> c3
    macs_v3 = 2 * c3 * c2 * 5 * 5 * (h//4) * (w//4)
    
    # v4: c3 -> c4
    macs_v4 = 2 * c4 * c3 * 5 * 5 * (h//4) * (w//4)
    
    # v5-v8: c4 -> c4 (4 layers)
    macs_v5v8 = 4 * 2 * c4 * c4 * 5 * 5 * (h//8) * (w//8)
    
    # v9: c4 -> c5
    macs_v9 = 2 * c5 * c4 * 15 * 15 * 1 * 1
    
    # seg_mask: c5 -> 1
    macs_seg = 2 * 1 * c5 * 1 * 1 * 1 * 1
    
    # extractor
    macs_ext1 = 2 * 8 * (c5 + 1) * 5 * 5 * 1 * 1
    macs_ext2 = 2 * 16 * 8 * 5 * 5 * 1 * 1
    macs_ext3 = 2 * 32 * 16 * 5 * 5 * 1 * 1
    
    total_macs = macs_v1 + macs_v2 + macs_v3 + macs_v4 + macs_v5v8 + macs_v9 + macs_seg + macs_ext1 + macs_ext2 + macs_ext3
    
    return total_macs / 1e9  # G MACs


def get_model_channels(model_name):
    """Get channel configuration for each model by reading from checkpoint"""
    # Map model name to file
    model_file_map = {
        'baseline': 'baseline_best.pth',
        'naive_l1_r20': 'naive_l1_r20.pth',
        'naive_l1_r40': 'naive_l1_r40.pth',
        'naive_l1_r60': 'naive_l1_r60.pth',
        'naive_fn_r20': 'naive_fn_r20.pth',
        'naive_fn_r40': 'naive_fn_r40.pth',
        'naive_fn_r60': 'naive_fn_r60.pth',
        'ours_c1_r20': 'ours_c1_r20.pth',
        'ours_c1_r40': 'ours_c1_r40.pth',
        'ours_c1_r60': 'ours_c1_r60.pth',
        'ours_c2_r20': 'ours_c2_r20.pth',
        'ours_c2_r40': 'ours_c2_r40.pth',
        'ours_c2_r60': 'ours_c2_r60.pth',
        'ours_c3_r20': 'ours_c3_r20.pth',
        'ours_c3_r40': 'ours_c3_r40.pth',
        'ours_c3_r60': 'ours_c3_r60.pth',
        'ours_c4_r20': 'ours_c4_r20.pth',
        'ours_c4_r40': 'ours_c4_r40.pth',
        'ours_c4_r60': 'ours_c4_r60.pth',
    }
    
    model_file = model_file_map.get(model_name)
    if model_file:
        config = get_model_config_from_ckpt(model_file)
        if config:
            return config
    
    # Default fallback
    return {'c1': 32, 'c2': 64, 'c3': 64, 'c4': 64, 'c5': 1024}


def main():
    # Models to profile
    models = [
        'baseline',
        'naive_l1_r20',
        'naive_l1_r40',
        'naive_l1_r60',
        'naive_fn_r20',
        'naive_fn_r40',
        'naive_fn_r60',
        'ours_c1_r20',
        'ours_c1_r40',
        'ours_c1_r60',
        'ours_c2_r20',
        'ours_c2_r40',
        'ours_c2_r60',
        'ours_c3_r20',
        'ours_c3_r40',
        'ours_c3_r60',
        'ours_c4_r20',
        'ours_c4_r40',
        'ours_c4_r60',
    ]
    
    results = []
    
    # Get baseline values
    baseline_channels = get_model_channels('baseline')
    baseline_params = calculate_params(baseline_channels)
    baseline_macs = calculate_macs(baseline_channels)
    
    print("\n" + "="*80)
    print("Model Profiling Results (Theoretical Calculation)")
    print("="*80)
    print(f"{'Model':<25} {'Params (M)':<15} {'MACs (G)':<15} {'Sparsity (%)':<15}")
    print("-"*80)
    
    for name in models:
        channels = get_model_channels(name)
        params = calculate_params(channels)
        macs = calculate_macs(channels)
        
        sparsity = (1 - params / baseline_params) * 100
        macs_reduction = (1 - macs / baseline_macs) * 100
        
        print(f"{name:<25} {params:<15.2f} {macs:<15.2f} {sparsity:<15.1f}")
        
        results.append({
            'model': name,
            'params_m': round(params, 2),
            'macs_g': round(macs, 2),
            'sparsity_pct': round(sparsity, 1),
            'macs_reduction_pct': round(macs_reduction, 1),
            'channels': channels
        })
    
    print("="*80)
    print(f"\nBaseline: {baseline_params:.2f} M params, {baseline_macs:.2f} G MACs")
    
    # Save results
    import json
    with open('profiling_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to profiling_results.json")


if __name__ == '__main__':
    main()
