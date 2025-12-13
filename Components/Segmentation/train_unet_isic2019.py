# train_unet_isic2019.py (FIXED VERSION)
"""
Train U-Net on ISIC-2019 with SAM-generated masks
Using segmentation_models_pytorch library
Optimized for Windows + PyCharm + CUDA
FIXED: Compatible with newer library versions
"""

import os
import json
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 80)
print("ISIC-2019 U-Net Training with Segmentation Models PyTorch")
print("=" * 80)


# ============================================================
# Configuration
# ============================================================
class Config:
    # Dataset Paths
    TRAIN_IMG_DIR = "../../Dataset/ISIC2019/train/images"
    TRAIN_MASK_DIR = "../../Dataset/ISIC2019/train/masks"
    VAL_IMG_DIR = "../../Dataset/ISIC2019/val/images"
    VAL_MASK_DIR = "../../Dataset/ISIC2019/val/masks"

    # Model Architecture
    ARCHITECTURE = "Unet"
    ENCODER = "resnet34"
    ENCODER_WEIGHTS = "imagenet"

    # Training Hyperparameters
    BATCH_SIZE = 4  # FIXED: Reduced for 4GB VRAM (RTX 3050 Ti)
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Image Processing
    IMG_HEIGHT = 256
    IMG_WIDTH = 256

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4

    # Saving
    CHECKPOINT_DIR = ""
    MODEL_NAME = "unet_pretrained.pth"
    LOG_DIR = "../../training_logs"
    PLOT_DIR = "../../training_plots"

    # Logging
    LOG_FILE = "training_log.json"
    SAVE_PLOTS = True
    PLOT_FREQUENCY = 5

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10


# Create directories
for directory in [Config.CHECKPOINT_DIR, Config.LOG_DIR, Config.PLOT_DIR]:
    os.makedirs(directory, exist_ok=True)


# ============================================================
# Dataset Class
# ============================================================
class ISICSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.images = []
        for ext in ['.jpg', '.jpeg', '.png']:
            self.images.extend([f for f in os.listdir(images_dir) if f.lower().endswith(ext)])

        self.valid_images = []
        for img_name in self.images:
            base_name = os.path.splitext(img_name)[0]
            mask_name = f"{base_name}_segmentation.png"
            mask_path = os.path.join(masks_dir, mask_name)

            if os.path.exists(mask_path):
                self.valid_images.append((img_name, mask_name))

        print(f"  Loaded {len(self.valid_images)} valid image-mask pairs from {images_dir}")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_name, mask_name = self.valid_images[idx]

        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = mask.unsqueeze(0)
        return image, mask


# ============================================================
# Data Augmentation (FIXED)
# ============================================================
def get_training_augmentation():
    return A.Compose([
        A.Resize(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            scale_limit=0.15,
            rotate_limit=45,
            shift_limit=0.1,
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT
        ),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=0.3, p=0.5),  # FIXED: Removed shift_limit
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
            A.CLAHE(clip_limit=4.0, p=1),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(std=(10.0, 50.0), p=1),  # FIXED: Changed var_limit to std
            A.GaussianBlur(blur_limit=(3, 7), p=1),
        ], p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_validation_augmentation():
    return A.Compose([
        A.Resize(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


# ============================================================
# Loss Function
# ============================================================
class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_bce=0.3, weight_focal=0.2):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.weight_focal = weight_focal
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        return 1 - dice_score

    def focal_loss(self, inputs, targets, alpha=0.25, gamma=2):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()

    def forward(self, inputs, targets):
        loss_dice = self.dice_loss(inputs, targets)
        loss_bce = self.bce(inputs, targets)
        loss_focal = self.focal_loss(inputs, targets)
        return (self.weight_dice * loss_dice +
                self.weight_bce * loss_bce +
                self.weight_focal * loss_focal)


# ============================================================
# Metrics
# ============================================================
def calculate_metrics(predictions, targets, threshold=0.5):
    preds = (predictions > threshold).float()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)

    dice = (2 * intersection + 1e-7) / (preds.sum() + targets.sum() + 1e-7)

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()

    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    recall = (tp + 1e-7) / (tp + fn + 1e-7)

    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }


# ============================================================
# Training Functions
# ============================================================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    metrics_sum = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0}

    pbar = tqdm(dataloader, desc="Training", leave=False, colour="green")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = torch.sigmoid(outputs)
            batch_metrics = calculate_metrics(preds, masks)
            epoch_loss += loss.item()
            for key in metrics_sum:
                metrics_sum[key] += batch_metrics[key]

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{batch_metrics["iou"]:.4f}'
        })

    num_batches = len(dataloader)
    avg_metrics = {
        'loss': epoch_loss / num_batches,
        'iou': metrics_sum['iou'] / num_batches,
        'dice': metrics_sum['dice'] / num_batches,
        'precision': metrics_sum['precision'] / num_batches,
        'recall': metrics_sum['recall'] / num_batches
    }
    return avg_metrics


def validate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    metrics_sum = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0}

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False, colour="cyan")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            preds = torch.sigmoid(outputs)
            batch_metrics = calculate_metrics(preds, masks)

            epoch_loss += loss.item()
            for key in metrics_sum:
                metrics_sum[key] += batch_metrics[key]

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{batch_metrics["iou"]:.4f}'
            })

    num_batches = len(dataloader)
    avg_metrics = {
        'loss': epoch_loss / num_batches,
        'iou': metrics_sum['iou'] / num_batches,
        'dice': metrics_sum['dice'] / num_batches,
        'precision': metrics_sum['precision'] / num_batches,
        'recall': metrics_sum['recall'] / num_batches
    }
    return avg_metrics


# ============================================================
# Visualization
# ============================================================
def save_predictions(model, dataloader, device, epoch, save_dir):
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(dataloader))
        images = images[:4].to(device)
        masks = masks[:4]
        outputs = torch.sigmoid(model(images))

        fig, axes = plt.subplots(4, 3, figsize=(12, 16))

        for i in range(4):
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)

            mask_true = masks[i, 0].cpu().numpy()
            mask_pred = outputs[i, 0].cpu().numpy()

            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(mask_true, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(mask_pred, cmap='gray')
            axes[i, 2].set_title(f'Prediction (Epoch {epoch})')
            axes[i, 2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'predictions_epoch_{epoch:03d}.png')
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved predictions to: {save_path}")


# ============================================================
# Main Training Loop
# ============================================================
def main():
    print(f"\nConfiguration:")
    print(f"  Device: {Config.DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    print(f"  Architecture: {Config.ARCHITECTURE}")
    print(f"  Encoder: {Config.ENCODER}")
    print(f"  Image Size: {Config.IMG_HEIGHT}x{Config.IMG_WIDTH}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Epochs: {Config.NUM_EPOCHS}")
    print(f"  Learning Rate: {Config.LEARNING_RATE}")
    print()

    # Load Datasets
    print("Loading datasets...")
    train_dataset = ISICSegmentationDataset(
        Config.TRAIN_IMG_DIR,
        Config.TRAIN_MASK_DIR,
        transform=get_training_augmentation()
    )

    val_dataset = ISICSegmentationDataset(
        Config.VAL_IMG_DIR,
        Config.VAL_MASK_DIR,
        transform=get_validation_augmentation()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print()

    # Create Model
    print("Creating model...")
    if Config.ARCHITECTURE == "Unet":
        model = smp.Unet(
            encoder_name=Config.ENCODER,
            encoder_weights=Config.ENCODER_WEIGHTS,
            in_channels=3,
            classes=1,
            activation=None
        )

    model = model.to(Config.DEVICE)
    print(f"  Model created: {Config.ARCHITECTURE} with {Config.ENCODER} encoder")
    print()

    # Setup Training
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
        # FIXED: Removed verbose parameter
    )

    # Training Loop
    print("Starting training...")
    print("=" * 80)

    best_iou = 0
    patience_counter = 0
    training_start_time = time.time()

    log_path = os.path.join(Config.LOG_DIR, Config.LOG_FILE)
    with open(log_path, 'w') as f:
        pass

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        epoch_start_time = time.time()

        print(f"\nEpoch {epoch}/{Config.NUM_EPOCHS}")
        print("-" * 80)

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
        val_metrics = validate(model, val_loader, criterion, Config.DEVICE)

        scheduler.step(val_metrics['iou'])

        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch} completed in {epoch_time / 60:.1f} minutes")
        print(f"Train → Loss: {train_metrics['loss']:.4f} | "
              f"IoU: {train_metrics['iou']:.4f} | "
              f"Dice: {train_metrics['dice']:.4f}")
        print(f"Val   → Loss: {val_metrics['loss']:.4f} | "
              f"IoU: {val_metrics['iou']:.4f} | "
              f"Dice: {val_metrics['dice']:.4f}")

        log_entry = {
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time,
            'timestamp': datetime.now().isoformat()
        }
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        if Config.SAVE_PLOTS and epoch % Config.PLOT_FREQUENCY == 0:
            save_predictions(model, val_loader, Config.DEVICE, epoch, Config.PLOT_DIR)

        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            patience_counter = 0
            save_path = os.path.join(Config.CHECKPOINT_DIR, Config.MODEL_NAME)
            torch.save(model.state_dict(), save_path)
            print(f"✓ Saved best model | IoU: {best_iou:.4f} → {save_path}")
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best IoU: {best_iou:.4f}")
            break

    total_time = time.time() - training_start_time

    print("\n" + "=" * 80)
    print("✓ Training Complete!")
    print("=" * 80)
    print(f"\nTraining Summary:")
    print(f"  Total time: {total_time / 3600:.1f} hours")
    print(f"  Best IoU: {best_iou:.4f}")
    print(f"  Model saved: {os.path.join(Config.CHECKPOINT_DIR, Config.MODEL_NAME)}")
    print(f"  Training log: {log_path}")
    print(f"  Prediction plots: {Config.PLOT_DIR}/")
    print("\n" + "=" * 80)
    print("✓ Ready to use in your pipeline!")
    print("  The model will be automatically loaded by your Main_Pipeline.py")
    print("=" * 80)


if __name__ == "__main__":
    main()