"""
Evaluate all models to get real performance metrics
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pickle
import time
from sklearn.metrics import f1_score, recall_score, precision_score

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


class KSDD2Dataset(torch.utils.data.Dataset):
    """KSDD2 Dataset for evaluation"""
    def __init__(self, root_dir, split='val'):
        self.root_dir = root_dir
        self.labels = []
        
        with open("splits/KSDD2/split_246.pyb", "rb") as f:
            train_samples, test_samples = pickle.load(f)
            if split == 'val':
                samples = test_samples[len(test_samples)//2:]
            elif split == 'test':
                samples = test_samples
        
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
    
    # Get channel config
    config = get_channel_config(state_dict)
    
    # Check if pruned
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


@torch.no_grad()
def evaluate_model(model, data_loader, device):
    """Evaluate model on test set"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_seg_preds = []
    all_seg_targets = []
    
    for images, masks, labels in data_loader:
        images = images.to(device)
        decision, seg_mask = model(images)
        
        pred_labels = (torch.sigmoid(decision.view(-1)) > 0.5).cpu().numpy()
        all_preds.extend(pred_labels)
        all_labels.extend(labels.numpy())
        
        # For segmentation
        seg_pred = torch.sigmoid(seg_mask)
        all_seg_preds.append(seg_pred.cpu().numpy())
        all_seg_targets.append(masks.numpy())
    
    # Calculate metrics
    recall = recall_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds) * 100
    
    # FNR = 100 - Recall
    fnr = 100 - recall
    
    return {
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'fnr': fnr
    }


@torch.no_grad()
def measure_latency(model, device, n_warmup=10, n_runs=100):
    """Measure inference latency"""
    model.eval()
    
    dummy_input = torch.randn(1, 3, 640, 232).to(device)
    
    # Warmup
    for _ in range(n_warmup):
        _ = model(dummy_input)
    
    # Measure
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(n_runs):
        _ = model(dummy_input)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / n_runs * 1000  # ms
    
    return avg_latency


@torch.no_grad()
def measure_throughput(model, device, batch_sizes=[1, 4, 8, 16, 32], n_runs=50):
    """Measure throughput at different batch sizes"""
    model.eval()
    
    results = {}
    
    for bs in batch_sizes:
        dummy_input = torch.randn(bs, 3, 640, 232).to(device)
        
        # Warmup
        for _ in range(5):
            _ = model(dummy_input)
        
        # Measure
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(n_runs):
            _ = model(dummy_input)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        total_images = n_runs * bs
        fps = total_images / total_time
        
        results[bs] = fps
    
    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = KSDD2Dataset('./datasets/KSDD2/test/', split='test')
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=2)
    print(f"Test samples: {len(test_dataset)}")
    
    # Models to evaluate
    models_to_eval = [
        'baseline_best.pth',
        'naive_l1_r20.pth',
        'naive_l1_r40.pth',
        'naive_l1_r60.pth',
        'naive_fn_r20.pth',
        'naive_fn_r40.pth',
        'naive_fn_r60.pth',
        'ours_c1_r20.pth',
        'ours_c1_r40.pth',
        'ours_c1_r60.pth',
        'ours_c2_r20.pth',
        'ours_c2_r40.pth',
        'ours_c2_r60.pth',
        'ours_c3_r20.pth',
        'ours_c3_r40.pth',
        'ours_c3_r60.pth',
        'ours_c4_r20.pth',
        'ours_c4_r40.pth',
        'ours_c4_r60.pth',
    ]
    
    results = []
    
    print("\n" + "="*60)
    print("Evaluating Models")
    print("="*60)
    
    for model_path in models_to_eval:
        if not os.path.exists(model_path):
            print(f"\nSkipping {model_path} (not found)")
            continue
        
        print(f"\nEvaluating {model_path}...")
        
        # Load model
        model = load_model(model_path, device)
        
        # Evaluate
        metrics = evaluate_model(model, test_loader, device)
        print(f"  Recall: {metrics['recall']:.2f}%, F1: {metrics['f1']:.2f}%, Precision: {metrics['precision']:.2f}%")
        
        # Measure latency
        latency = measure_latency(model, device)
        print(f"  Latency: {latency:.2f} ms")
        
        # Measure throughput
        throughput = measure_throughput(model, device)
        print(f"  Throughput (bs=1): {throughput[1]:.1f} FPS")
        
        # Get params
        n_params = sum(p.numel() for p in model.parameters())
        
        results.append({
            'model': model_path.replace('.pth', ''),
            'params': n_params,
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'fnr': metrics['fnr'],
            'latency': latency,
            'throughput_bs1': throughput[1],
        })
    
    # Save results
    import json
    with open('real_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print("\nResults saved to real_evaluation_results.json")
    
    # Print summary
    print("\nSummary:")
    for r in results:
        print(f"  {r['model']}: Recall={r['recall']:.1f}%, F1={r['f1']:.1f}%, Latency={r['latency']:.1f}ms")


if __name__ == '__main__':
    main()
