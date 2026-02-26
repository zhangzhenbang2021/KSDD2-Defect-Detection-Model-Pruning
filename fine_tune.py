"""
Fine-tuning Script for SegDecNet
This script fine-tunes the baseline model (or any model checkpoint)
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
from sklearn.metrics import f1_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SegDecNet


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class KSDD2Dataset(torch.utils.data.Dataset):
    """KSDD2 Dataset"""
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.labels = []
        
        import pickle
        fn = f"KSDD2/split_246.pyb"
        with open(f"splits/{fn}", "rb") as f:
            train_samples, test_samples = pickle.load(f)
            if split == 'train':
                samples = train_samples
            elif split == 'val':
                samples = test_samples[:len(test_samples)//2]
            elif split == 'test':
                samples = test_samples[len(test_samples)//2:]
        
        for part, is_segmented in samples:
            img_path = os.path.join(root_dir, str(part) + ".png")
            img = Image.open(img_path).convert("RGB")
            
            mask_path = os.path.join(root_dir, part + "_label.png")
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")
                mask = np.array(mask) / 255.0
            else:
                mask = np.zeros((img.size[1], img.size[0]))
            
            img = img.resize((232, 640), Image.BILINEAR)
            mask = mask.resize((232, 640), Image.NEAREST)
            
            img_tensor = transforms.ToTensor()(img)
            img_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)
            mask_tensor = transforms.ToTensor()(mask)
            
            # Determine label by analyzing actual mask content (not is_segmented flag)
            label = 1 if np.sum(mask > 0) > 0 else 0
            self.labels.append((img_tensor, mask_tensor, torch.tensor(label, dtype=torch.float32)))
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.labels[idx]


class FineTuner:
    """Fine-tuner for model"""
    
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
        
        if self.config.get('weighted_seg_loss', True):
            p = self.config.get('weighted_seg_loss_p', 2)
            max_w = self.config.get('weighted_seg_loss_max', 3)
            
            seg_weights = torch.where(masks > 0.5, 
                                      torch.tensor(max_w, device=masks.device), 
                                      torch.tensor(1.0, device=masks.device))
            seg_weights = seg_weights ** p
            seg_weights = torch.clamp(seg_weights, 1.0, max_w)
            
            seg_loss = self.seg_loss_fn(seg_mask, masks)
            seg_loss = (seg_loss * seg_weights).mean()
        else:
            seg_loss = self.seg_loss_fn(seg_mask, masks).mean()
        
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
        return f1, recall
    
    def train(self, num_epochs, save_name):
        best_f1 = 0
        best_recall = 0
        
        for epoch in range(num_epochs):
            self.model.set_gradient_multipliers(0.0)
            
            train_loss, train_acc = self.train_epoch(epoch)
            val_f1, val_recall = self.evaluate()
            
            self.scheduler.step(val_f1)
            
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val F1: {val_f1:.4f}, Recall: {val_recall:.4f}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_recall = val_recall
                torch.save(self.model.state_dict(), save_name)
                print(f"  -> Saved best model with F1: {best_f1:.4f}")
        
        return best_f1, best_recall


def run_fine_tuning(checkpoint_path, num_epochs=15, learning_rate=0.001):
    """Run fine-tuning on a checkpoint"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model from checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    model = SegDecNet(device, 232, 640, 3)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    model = model.to(device)
    model.set_gradient_multipliers(1.0)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    
    # Load dataset
    print("\nLoading datasets...")
    train_dataset = KSDD2Dataset('./datasets/KSDD2/', split='train')
    val_dataset = KSDD2Dataset('./datasets/KSDD2/', split='val')
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=4)
    
    # Fine-tune
    config = {
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'delta_cls_loss': 1.0,
        'weighted_seg_loss': True,
        'weighted_seg_loss_p': 2,
        'weighted_seg_loss_max': 3,
        'dyn_balanced_loss': True,
        'gradient_adjustment': True,
    }
    
    print("\nStarting fine-tuning...")
    fine_tuner = FineTuner(model, train_loader, val_loader, device, config)
    save_name = checkpoint_path.replace('.pth', '_finetuned.pth')
    best_f1, best_recall = fine_tuner.train(config['num_epochs'], save_name)
    
    print(f"\n=== Fine-tuning Complete ===")
    print(f"Best F1: {best_f1:.4f}, Best Recall: {best_recall:.4f}")
    print(f"Saved to: {save_name}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='baseline_best.pth')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    set_seed(42)
    run_fine_tuning(args.checkpoint, args.epochs, args.lr)
