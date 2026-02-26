"""
Pruning Framework for SegDecNet
Provides channel pruning based on L1-Norm and FeatureNorm (FN-gamma)
"""

import torch
import torch.nn as nn
import numpy as np
from models import SegDecNet, FeatureNorm


def get_conv_layers(model):
    """Get all Conv2d layer names and references"""
    conv_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers[name] = module
    return conv_layers


def get_featurenorm_layers(model):
    """Get all FeatureNorm layer names and references"""
    fn_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, FeatureNorm):
            fn_layers[name] = module
    return fn_layers


def compute_l1_importance(model):
    """
    Compute channel importance based on L1-Norm
    Returns: dict {layer_name: importance_scores}
    """
    importance = {}
    conv_layers = get_conv_layers(model)
    
    for name, module in conv_layers.items():
        # Get weight [out_channels, in_channels, kH, kW]
        weight = module.weight.data
        # Compute L1-norm for each output channel
        l1_norm = torch.norm(weight.view(weight.shape[0], -1), p=1, dim=1)
        importance[name] = l1_norm.cpu().numpy()
    
    return importance


def compute_featurenorm_importance(model):
    """
    Compute channel importance based on FeatureNorm gamma parameter
    Returns: dict {layer_name: importance_scores}
    """
    importance = {}
    fn_layers = get_featurenorm_layers(model)
    
    for name, module in fn_layers.items():
        # FeatureNorm's scale parameter is gamma
        gamma = module.scale.data.abs().squeeze()
        # Handle scalar gamma case (1,1,1,1)
        if gamma.dim() == 0:
            gamma = gamma.unsqueeze(0)
        importance[name] = gamma.cpu().numpy()
    
    return importance


def get_prune_indices(importance_dict, prune_rate):
    """
    Determine channel indices to keep based on importance
    
    Args:
        importance_dict: {layer_name: importance_scores}
        prune_rate: pruning rate (0-1)
    
    Returns:
        keep_indices_dict: {layer_name: indices_to_keep}
    """
    keep_indices_dict = {}
    
    for name, scores in importance_dict.items():
        scores = torch.tensor(scores)
        num_channels = scores.shape[0]
        num_keep = int(num_channels * (1 - prune_rate))
        
        # Keep channels with highest importance
        _, indices = torch.topk(scores, num_keep)
        keep_indices_dict[name] = indices.sort()[0]
    
    return keep_indices_dict


def create_pruned_model_from_indices(model, keep_indices_dict, device='cuda'):
    """
    Create a new pruned model based on retained channel indices
    
    Args:
        model: original model
        keep_indices_dict: {layer_name: indices_to_keep}
        device: device
    
    Returns:
        new model instance
    """
    # Create new model
    new_model = SegDecNet(device, model.input_width, model.input_height, model.input_channels)
    new_model = new_model.to(device)
    
    # Get original model's state dict
    old_state_dict = model.state_dict()
    new_state_dict = new_model.state_dict()
    
    # Iterate through all layers in original model
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if name in keep_indices_dict:
                keep_idx = keep_indices_dict[name]
                # Prune output channels of current conv layer
                old_key = f"{name}.weight"
                new_state_dict[old_key] = old_state_dict[old_key][keep_idx, :, :, :].clone()
        
        elif isinstance(module, FeatureNorm):
            # Find corresponding conv layer to determine which channels to keep
            # FeatureNorm follows Conv, names like "volume.1" correspond to "volume.0"
            # Need to find associated conv layer
            conv_name = name.replace('.0', '.0').replace('.1', '.0').replace('.3', '.2').replace('.5', '.4')
            if conv_name in keep_indices_dict:
                keep_idx = keep_indices_dict[conv_name]
                
                # Prune FeatureNorm parameters
                old_scale = old_state_dict[f"{name}.scale"]
                old_bias = old_state_dict[f"{name}.bias"]
                old_running_mean = old_state_dict[f"{name}.running_mean"]
                old_running_var = old_state_dict[f"{name}.running_var"]
                
                new_state_dict[f"{name}.scale"] = old_scale[keep_idx].clone()
                new_state_dict[f"{name}.bias"] = old_bias[keep_idx].clone()
                new_state_dict[f"{name}.running_mean"] = old_running_mean[keep_idx].clone()
                new_state_dict[f"{name}.running_var"] = old_running_var[keep_idx].clone()
    
    # Handle conv layers that need input channel adjustment
    # This requires more complex logic to determine input channel correspondence
    
    # Simplified: manually build new state dict
    new_state_dict = build_pruned_state_dict(model, keep_indices_dict)
    
    new_model.load_state_dict(new_state_dict)
    return new_model


def build_pruned_state_dict(model, keep_indices_dict):
    """
    Build pruned state dict, handling conv layer input/output channel correspondence
    """
    old_state_dict = model.state_dict()
    new_state_dict = {}
    
    # Define conv layer order and their connections
    # volume layers: Conv -> FeatureNorm -> Conv -> FeatureNorm ...
    conv_order = [
        ('volume.0', 3),       # First conv, input 3 channels
        ('volume.3', None),    # Second conv, input from previous
        ('volume.6', None),
        ('volume.9', None),
        ('volume.12', None),
        ('volume.15', None),
        ('volume.18', None),
        ('volume.21', None),
    ]
    
    # Track output channel count for each layer
    channel_map = {}  # layer_name -> actual_keep_indices
    
    # First process all FeatureNorm layers (determine channel counts)
    fn_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, FeatureNorm):
            fn_layers[name] = module
    
    # Build layer connection relationships
    # volume.0 (Conv) -> volume.1 (FN) -> volume.3 (Conv) -> volume.4 (FN) -> ...
    layer_connections = [
        ('volume.0', 'volume.1', 'volume.3'),  # (conv, fn, next_conv)
        ('volume.3', 'volume.4', 'volume.6'),
        ('volume.6', 'volume.7', 'volume.9'),
        ('volume.9', 'volume.10', 'volume.12'),
        ('volume.12', 'volume.13', 'volume.15'),
        ('volume.15', 'volume.16', 'volume.18'),
        ('volume.18', 'volume.19', 'volume.21'),
        ('volume.21', 'volume.22', 'seg_mask.0'),
    ]
    
    # Need to track which channels are retained
    # For first conv layer, use given keep_indices
    # For subsequent conv layers, determine based on previous layer's FN
    
    # Process first conv layer first
    if 'volume.0' in keep_indices_dict:
        channel_map['volume.0'] = keep_indices_dict['volume.0']
    
    # Process each connection
    processed_convs = set()
    
    for conv_name, fn_name, next_conv_name in layer_connections:
        if conv_name in keep_indices_dict:
            # Get output channels to keep for current conv layer
            keep_out = keep_indices_dict[conv_name]
            channel_map[conv_name] = keep_out
            processed_convs.add(conv_name)
            
            # Get input channels to keep for next conv layer
            # Input channels = current FN's output channels = current conv's output channels
            if fn_name in fn_layers:
                # Copy output channels from current conv layer to FN layer
                old_scale = old_state_dict[f"{fn_name}.scale"]
                old_bias = old_state_dict[f"{fn_name}.bias"]
                old_running_mean = old_state_dict[f"{fn_name}.running_mean"]
                old_running_var = old_state_dict[f"{fn_name}.running_var"]
                
                new_state_dict[f"{fn_name}.scale"] = old_scale[keep_out].clone()
                new_state_dict[f"{fn_name}.bias"] = old_bias[keep_out].clone()
                new_state_dict[f"{fn_name}.running_mean"] = old_running_mean[keep_out].clone()
                new_state_dict[f"{fn_name}.running_var"] = old_running_var[keep_out].clone()
            
            # Handle next conv layer's input channels
            if next_conv_name in model.state_dict():
                # Next conv layer needs to keep input channels = current conv's retained output channels
                next_conv_key = f"{next_conv_name}.weight"
                if next_conv_key in old_state_dict:
                    old_weight = old_state_dict[next_conv_key]
                    # Input channels = current retained output channels
                    new_weight = old_weight[:, keep_out, :, :].clone()
                    new_state_dict[next_conv_key] = new_weight
                    channel_map[next_conv_name] = keep_out
    
    # Handle layers that don't need pruning (copy original weights)
    for key in old_state_dict:
        if key not in new_state_dict:
            new_state_dict[key] = old_state_dict[key].clone()
    
    return new_state_dict


def prune_model_l1(model, prune_rate, device='cuda'):
    """
    Prune model using L1-Norm
    
    Args:
        model: original model
        prune_rate: pruning rate (0-1)
        device: device
    
    Returns:
        pruned model
    """
    # Compute L1 importance
    importance = compute_l1_importance(model)
    
    # Get retain indices
    keep_indices = get_prune_indices(importance, prune_rate)
    
    # Create pruned model
    pruned_model = create_pruned_model_from_indices(model, keep_indices, device)
    
    return pruned_model


def prune_model_featurenorm(model, prune_rate, device='cuda'):
    """
    Prune model using FeatureNorm gamma parameter
    
    Args:
        model: sparse model (trained with L1 regularization)
        prune_rate: pruning rate (0-1)
        device: device
    
    Returns:
        pruned model
    """
    # Compute FeatureNorm importance
    importance = compute_featurenorm_importance(model)
    
    # Get retain indices
    keep_indices = get_prune_indices(importance, prune_rate)
    
    # Create pruned model
    pruned_model = create_pruned_model_from_indices(model, keep_indices, device)
    
    return pruned_model


def count_parameters(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters())


def print_model_channels(model):
    """Print channel counts for each layer"""
    print("=== Model Channel Configuration ===")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"{name}: in={module.in_channels}, out={module.out_channels}")
        elif isinstance(module, FeatureNorm):
            print(f"{name}: num_features={module.scale.shape[1]}")
    print("===================================")
