"""
C4: Taylor-FO (First-Order) Pruning for SegDecNet
An incremental improvement over Naive L1 pruning using gradient-weighted importance

Core Idea:
1. Run forward pass with real data to get Loss
2. Run backward pass to compute gradients (without updating weights)
3. Compute importance: I_c = sum(|W_c * grad(W_c)|)
4. Channels with low Taylor-FO scores contribute less to the loss and can be pruned
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


class KSDD2Dataset(torch.utils.data.Dataset):
    """KSDD2 Dataset for pruning fine-tuning"""
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.labels = []
        
        with open("splits/KSDD2/split_246.pyb", "rb") as f:
            train_samples, test_samples = pickle.load(f)
            if split == 'train':
                samples = train_samples
            elif split == 'val':
                samples = test_samples[len(test_samples)//2:]
            elif split == 'test':
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
            
            # Determine label by analyzing actual mask content
            label = 1 if np.sum(mask > 0) > 0 else 0
            self.labels.append((img_tensor, mask_tensor, torch.tensor(label, dtype=torch.float32)))
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.labels[idx]


def compute_taylor_importance(model, dataloader, device, config, num_batches=10):
    """
    Method C4: Compute Taylor First-Order (Gradient-Weighted L1) importance
    
    Core Idea:
    - Run forward pass with real data to get Loss
    - Run backward pass to compute gradients (without updating weights)
    - Compute importance: I_c = sum(|W_c * grad(W_c)|)
    - Channels with low Taylor-FO scores contribute less to the loss
    
    Formula: I_c = sum(|W_{c,i,j,k} * grad(W_{c,i,j,k})|)
    """
    print("  -> Running Taylor-FO Calibration (computing gradients)...")
    
    # Must be in train mode for gradients to flow properly
    model.train()
    model.zero_grad()
    
    # Loss functions (same as used in fine-tuning)
    seg_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    dec_loss_fn = nn.BCEWithLogitsLoss()
    
    batch_count = 0
    for i, (images, masks, labels) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        
        # Forward pass
        decision, seg_mask_pred = model(images)
        
        # Classification loss
        dec_loss = dec_loss_fn(decision.view(-1), labels.view(-1))
        
        # Segmentation loss (resize to match target)
        seg_mask_resized = torch.nn.functional.interpolate(
            seg_mask_pred, size=masks.shape[-2:], mode='bilinear', align_corners=False
        )
        
        if config.get('weighted_seg_loss', True):
            p = config.get('weighted_seg_loss_p', 2)
            max_w = config.get('weighted_seg_loss_max', 3)
            seg_weights = torch.where(masks > 0.5, 
                                      torch.tensor(max_w, device=device, dtype=torch.float), 
                                      torch.tensor(1.0, device=device, dtype=torch.float))
            seg_weights = seg_weights ** p
            seg_weights = torch.clamp(seg_weights, 1.0, max_w)
            seg_loss = (seg_loss_fn(seg_mask_resized, masks) * seg_weights).mean()
        else:
            seg_loss = seg_loss_fn(seg_mask_resized, masks).mean()
        
        # Combined loss (simplified weights)
        total_loss = seg_loss + dec_loss
        
        # Backward pass to get gradients (IMPORTANT: no optimizer.step()!)
        total_loss.backward()
        
        batch_count += 1
        if batch_count >= num_batches:
            break
    
    # Extract parameters and gradients, compute Taylor scores
    importance = {}
    conv_layers = ['volume.0.0', 'volume.2.0', 'volume.3.0', 'volume.4.0',
                   'volume.6.0', 'volume.7.0', 'volume.8.0', 'volume.9.0', 'volume.11.0']
    
    # Get all parameters by name
    params_dict = dict(model.named_parameters())
    
    for conv_name in conv_layers:
        weight_key = f'{conv_name}.weight'
        if weight_key in params_dict:
            w = params_dict[weight_key]
            grad = w.grad
            
            if grad is None:
                raise RuntimeError(f"No gradient found for {weight_key}. Calibration failed.")
            
            # Core formula: |W * grad(W)|
            # Shape: [out_channels, in_channels, kH, kW]
            score_matrix = torch.abs(w * grad)
            
            # Sum over all dimensions except output channel (dim=0)
            taylor_score = torch.sum(score_matrix, dim=(1, 2, 3))
            
            importance[conv_name] = taylor_score.detach().cpu().numpy()
    
    # Clean up gradients to keep model clean
    model.zero_grad()
    
    return importance


def get_keep_indices(importance_dict, prune_rate):
    """Get indices to keep based on importance (higher = more important)"""
    keep_indices = {}
    for name, scores in importance_dict.items():
        scores_tensor = torch.tensor(scores)
        num_channels = scores_tensor.shape[0]
        num_keep = int(num_channels * (1 - prune_rate))
        num_keep = max(1, num_keep)  # At least keep 1 channel
        # Keep channels with highest importance
        _, indices = torch.topk(scores_tensor, num_keep)
        keep_indices[name] = indices.sort()[0].numpy()
    return keep_indices


def prune_and_build_pruned_model(model, keep_indices, device):
    """Build pruned model with correct architecture"""
    state_dict = model.state_dict()
    
    new_channels = {}
    for conv_name, indices in keep_indices.items():
        new_channels[conv_name] = len(indices)
    
    channel_config = {
        'c1': new_channels['volume.0.0'],
        'c2': new_channels['volume.2.0'],
        'c3': new_channels['volume.3.0'],
        'c4': new_channels['volume.4.0'],
        'c5': new_channels['volume.11.0'],
    }
    
    new_model = PrunedSegDecNet(device, 232, 640, 3, channel_config)
    new_model = new_model.to(device)
    pruned_state = new_model.state_dict()
    
    c1, c2, c3, c4, c5 = channel_config['c1'], channel_config['c2'], channel_config['c3'], channel_config['c4'], channel_config['c5']
    
    # Get keep indices for each layer
    keep_v1 = keep_indices['volume.0.0']
    keep_v2 = keep_indices['volume.2.0']
    keep_v3 = keep_indices['volume.3.0']
    keep_v4 = keep_indices['volume.4.0']
    keep_v5 = keep_indices['volume.6.0']
    keep_v6 = keep_indices['volume.7.0']
    keep_v7 = keep_indices['volume.8.0']
    keep_v8 = keep_indices['volume.9.0']
    keep_v9 = keep_indices['volume.11.0']
    
    # Map weights - use importance-based indices
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
    
    # Seg mask: input=c5, output=1
    pruned_state['seg_mask.0.weight'] = state_dict['seg_mask.0.weight'][:, keep_v9, :, :].clone()
    pruned_state['seg_mask.1.scale'] = state_dict['seg_mask.1.scale'].clone()
    
    # Extractor: adjust input channels
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
    
    # FC layer
    pruned_state['fc.weight'] = state_dict['fc.weight'].clone()
    pruned_state['fc.bias'] = state_dict['fc.bias'].clone()
    
    new_model.load_state_dict(pruned_state, strict=True)
    return new_model, channel_config


class FineTuner:
    """Fine-tuner for pruned model"""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        self.seg_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.dec_loss_fn = nn.BCEWithLogitsLoss()
        
        self.optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=1e-4)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        self.model.set_gradient_multipliers(0.0)
        
    def compute_loss(self, pred, seg_mask, labels, masks, epoch):
        total_epochs = self.config['num_epochs']
        
        if self.config.get('dyn_balanced_loss', True):
            seg_loss_weight = 1 - (epoch / total_epochs)
            dec_loss_weight = self.config.get('delta_cls_loss', 1.0) * (epoch / total_epochs)
        else:
            seg_loss_weight = 1.0
            dec_loss_weight = self.config.get('delta_cls_loss', 1.0)
        
        dec_loss = self.dec_loss_fn(pred.view(-1), labels.view(-1))
        
        seg_mask_resized = torch.nn.functional.interpolate(
            seg_mask, size=masks.shape[-2:], mode='bilinear', align_corners=False
        )
        
        if self.config.get('weighted_seg_loss', True):
            p = self.config.get('weighted_seg_loss_p', 2)
            max_w = self.config.get('weighted_seg_loss_max', 3)
            
            seg_weights = torch.where(masks > 0.5, 
                                      torch.tensor(max_w, device=masks.device), 
                                      torch.tensor(1.0, device=masks.device))
            seg_weights = seg_weights ** p
            seg_weights = torch.clamp(seg_weights, 1.0, max_w)
            
            seg_loss = self.seg_loss_fn(seg_mask_resized, masks)
            seg_loss = (seg_loss * seg_weights).mean()
        else:
            seg_loss = self.seg_loss_fn(seg_mask_resized, masks).mean()
        
        total_loss = seg_loss_weight * seg_loss + dec_loss_weight * dec_loss
        return total_loss, seg_loss, dec_loss
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, masks, labels in self.train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            decision, seg_mask_pred = self.model(images)
            
            loss, _, _ = self.compute_loss(decision, seg_mask_pred, labels, masks, epoch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred_labels = (torch.sigmoid(decision.view(-1)) > 0.5).float()
            correct += (pred_labels == labels.view(-1)).sum().item()
            total += labels.size(0)
        
        return total_loss / len(self.train_loader), correct / total
    
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
            self.model.set_gradient_multipliers(0.0)
            
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


def run_c4_pruning(prune_rates=[0.2, 0.4, 0.6]):
    """Run C4 (Taylor-FO) pruning with fine-tuning"""
    
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
    
    # Load datasets
    print("\n" + "="*60)
    print("Loading datasets...")
    print("="*60)
    train_dataset = KSDD2Dataset('./datasets/KSDD2/train/', split='train')
    val_dataset = KSDD2Dataset('./datasets/KSDD2/test/', split='val')
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=2)
    
    # Fine-tuning config (used for Taylor importance computation)
    fine_tune_config = {
        'learning_rate': 0.001,
        'num_epochs': 15,
        'delta_cls_loss': 1.0,
        'weighted_seg_loss': True,
        'weighted_seg_loss_p': 2,
        'weighted_seg_loss_max': 3,
        'dyn_balanced_loss': True,
    }
    
    # ========== Compute Taylor-FO importance ONCE outside the loop ==========
    # The importance scores are independent of prune_rate
    print("\n" + "="*60)
    print("Step 1: Computing Taylor-FO Importance (Gradient-Weighted L1)")
    print("="*60)
    importance = compute_taylor_importance(model, train_loader, device, fine_tune_config, num_batches=10)
    
    for name, scores in importance.items():
        print(f"  {name}: {len(scores)} channels, mean Taylor={np.mean(scores):.6f}")
    
    # Prune each rate - using pre-computed importance
    for prune_rate in prune_rates:
        print("\n" + "="*60)
        print(f"C4 (Taylor-FO) Pruning with rate: {prune_rate*100:.0f}%")
        print("="*60)
        
        # Get keep indices based on Taylor importance
        keep_indices = get_keep_indices(importance, prune_rate)
        
        # Build pruned model
        print("\nStep 2: Building pruned model with Taylor-FO channel selection...")
        pruned_model, channel_config = prune_and_build_pruned_model(model, keep_indices, device)
        pruned_model.set_gradient_multipliers(1.0)
        
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        print(f"  Pruned parameters: {pruned_params:,} ({pruned_params/orig_params*100:.1f}%)")
        print(f"  Channel config: c1={channel_config['c1']}, c2={channel_config['c2']}, c3={channel_config['c3']}, c4={channel_config['c4']}, c5={channel_config['c5']}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 640, 232).to(device)
        decision, seg_mask = pruned_model(dummy_input)
        print(f"  Forward pass OK! decision={decision.shape}, seg_mask={seg_mask.shape}")
        
        # Fine-tuning
        print(f"\nStep 3: Fine-tuning for {fine_tune_config['num_epochs']} epochs...")
        fine_tuner = FineTuner(pruned_model, train_loader, val_loader, device, fine_tune_config)
        
        save_name = f'ours_c4_r{int(prune_rate*100)}.pth'
        best_f1, best_recall = fine_tuner.train(fine_tune_config['num_epochs'], save_name)
        
        print(f"\n=== Result for r={prune_rate*100:.0f}%: F1={best_f1:.4f}, Recall={best_recall:.4f} ===")
        print(f"  Saved to: {save_name}")
    
    print("\n" + "="*60)
    print("C4 (Taylor-FO) Pruning Complete!")
    print("="*60)
    
    # Summary
    print("\n=== Summary ===")
    print(f"Baseline: {orig_params:,} params")
    for rate in prune_rates:
        save_name = f'ours_c4_r{int(rate*100)}.pth'
        print(f"Ours C4 r{int(rate*100)}: {save_name}")


if __name__ == '__main__':
    set_seed(42)
    run_c4_pruning([0.2, 0.4, 0.6])
