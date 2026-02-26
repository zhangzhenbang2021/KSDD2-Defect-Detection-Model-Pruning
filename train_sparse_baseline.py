"""
Sparse Baseline Training Script (Network Slimming)
Trains a sparse model with L1 regularization on FeatureNorm gamma
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import random
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SegDecNet, FeatureNorm


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class KSDD2Dataset(torch.utils.data.Dataset):
    """KSDD2 Dataset"""
    def __init__(self, root_dir, split='train', transform=None, target_size=(512, 512)):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_size = target_size
        
        # Load split file
        split_file = "splits/KSDD2/split_246.pyb"
        
        with open(split_file, "rb") as f:
            train_samples, test_samples = pickle.load(f)
            if split == 'train':
                samples = train_samples
                data_folder = 'train'
            elif split == 'val':
                # Use second half of test set for validation
                samples = test_samples[len(test_samples)//2:]
                data_folder = 'test'
            elif split == 'test':
                samples = test_samples[len(test_samples)//2:]
                data_folder = 'test'
        
        self.labels = []
        
        for part, is_segmented in samples:
            # Load from appropriate folder
            img_path = os.path.join(root_dir, data_folder, str(part) + ".png")
            if not os.path.exists(img_path):
                continue
                
            img = Image.open(img_path).convert("RGB")
            
            mask_path = os.path.join(root_dir, data_folder, str(part) + "_GT.png")
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


class SparseTrainer:
    """Sparse Trainer - Adds L1 regularization on FeatureNorm gamma"""
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Loss functions
        self.seg_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.dec_loss_fn = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        # Gradient multiplier
        self.model.set_gradient_multipliers(0.0)
        
    def compute_sparse_loss(self, epoch):
        """
        Compute loss with L1 regularization
        L1 regularization applied on FeatureNorm gamma (scale parameter)
        """
        total_epochs = self.config['num_epochs']
        
        # Dynamic loss weights
        if self.config.get('dyn_balanced_loss', True):
            seg_loss_weight = 1 - (epoch / total_epochs)
            dec_loss_weight = self.config.get('delta_cls_loss', 1.0) * (epoch / total_epochs)
        else:
            seg_loss_weight = 1.0
            dec_loss_weight = self.config.get('delta_cls_loss', 1.0)
        
        # L1 regularization - applied on FeatureNorm gamma
        l1_regularization = 0.0
        lambda_factor = self.config.get('l1_lambda', 1e-4)
        
        for module in self.model.modules():
            if isinstance(module, FeatureNorm):
                l1_regularization += torch.sum(torch.abs(module.scale))
        
        return seg_loss_weight, dec_loss_weight, l1_regularization, lambda_factor
    
    def compute_loss(self, pred, seg_mask, labels, masks, epoch):
        """Compute total loss"""
        seg_loss_weight, dec_loss_weight, l1_reg, lambda_factor = self.compute_sparse_loss(epoch)
        
        # Classification loss
        dec_loss = self.dec_loss_fn(pred.view(-1), labels.view(-1))
        
        # Segmentation loss
        # Resize seg_mask to match target mask size
        seg_mask = torch.nn.functional.interpolate(
            seg_mask, size=masks.shape[2:], mode='bilinear', align_corners=False
        )
        
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
        
        # Total loss = seg loss + dec loss + L1 regularization
        total_loss = seg_loss_weight * seg_loss + dec_loss_weight * dec_loss + lambda_factor * l1_reg
        
        return total_loss, seg_loss, dec_loss, l1_reg
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_seg_loss = 0
        total_dec_loss = 0
        total_l1_loss = 0
        correct = 0
        total = 0
        
        for images, masks, labels in self.train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            decision, seg_mask_pred = self.model(images)
            
            loss, seg_loss, dec_loss, l1_loss = self.compute_loss(decision, seg_mask_pred, labels, masks, epoch)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_seg_loss += seg_loss.item()
            total_dec_loss += dec_loss.item()
            total_l1_loss += l1_loss.item()
            
            pred_labels = (torch.sigmoid(decision.view(-1)) > 0.5).float()
            correct += (pred_labels == labels.view(-1)).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        avg_seg_loss = total_seg_loss / len(self.train_loader)
        avg_dec_loss = total_dec_loss / len(self.train_loader)
        avg_l1_loss = total_l1_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, avg_seg_loss, avg_dec_loss, avg_l1_loss, accuracy
    
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
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        f1 = f1_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        
        return f1, recall
    
    def train(self, num_epochs, save_path):
        best_f1 = 0
        best_recall = 0
        
        for epoch in range(num_epochs):
            # Update gradient multipliers
            if self.config.get('gradient_adjustment', True):
                self.model.set_gradient_multipliers(0.0)
            
            # Train
            train_loss, train_seg_loss, train_dec_loss, train_l1_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_f1, val_recall = self.evaluate()
            
            # Update learning rate
            self.scheduler.step(val_f1)
            
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Loss: {train_loss:.4f} (Seg: {train_seg_loss:.4f}, Dec: {train_dec_loss:.4f}, L1: {train_l1_loss:.4f}) | "
                  f"Acc: {train_acc:.4f} | "
                  f"Val F1: {val_f1:.4f}, Recall: {val_recall:.4f}")
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_recall = val_recall
                save_file = os.path.join(save_path, 'baseline_slimming_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'f1': best_f1,
                    'recall': best_recall,
                }, save_file)
                print(f"  -> Saved best model with F1: {best_f1:.4f}")
        
        return best_f1, best_recall
    
    def print_featurenorm_scales(self):
        """Print FeatureNorm gamma values to verify sparsity"""
        print("\n=== FeatureNorm Gamma (Scale) Statistics ===")
        for name, module in self.model.named_modules():
            if isinstance(module, FeatureNorm):
                scales = module.scale.data.abs().squeeze()
                if scales.dim() == 0:
                    scales = scales.unsqueeze(0)
                print(f"{name}: min={scales.min():.4f}, max={scales.max():.4f}, mean={scales.mean():.4f}, zeros={(scales < 0.01).sum().item()}/{scales.numel()}")
        print("=============================================\n")


def train_sparse_baseline():
    """Train sparse baseline model"""
    
    # Configuration
    config = {
        'dataset_path': './datasets/KSDD2/',
        'batch_size': 10,
        'num_epochs': 15,
        'learning_rate': 0.01,
        'delta_cls_loss': 1.0,
        'weighted_seg_loss': True,
        'weighted_seg_loss_p': 2,
        'weighted_seg_loss_max': 3,
        'dyn_balanced_loss': True,
        'gradient_adjustment': True,
        'l1_lambda': 1e-4,  # L1 regularization coefficient
    }
    
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = KSDD2Dataset(config['dataset_path'], split='train')
    val_dataset = KSDD2Dataset(config['dataset_path'], split='val')
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    # Create model
    print("Creating SegDecNet model...")
    model = SegDecNet(device, 232, 640, 3)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create trainer
    print(f"Using gradient adjustment (gradient_multiplier=0)")
    print(f"L1 lambda: {config['l1_lambda']}")
    trainer = SparseTrainer(model, train_loader, val_loader, device, config)
    
    # Train
    save_path = '.'
    best_f1, best_recall = trainer.train(config['num_epochs'], save_path)
    
    # Print FeatureNorm gamma statistics
    trainer.print_featurenorm_scales()
    
    print(f"\nTraining completed!")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Best Recall: {best_recall:.4f}")
    print(f"Model saved to: {os.path.join(save_path, 'baseline_slimming_best.pth')}")


if __name__ == '__main__':
    train_sparse_baseline()
