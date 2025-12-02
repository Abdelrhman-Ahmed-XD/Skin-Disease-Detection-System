# Components/Segmentation/Unet_Segmentation_Algorithm.py
import os
import cv2
import torch
import torch.nn as nn
import numpy as np

# Print warning only once
_weights_warning_shown = False

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec4 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        d1 = self.up1(b);  d1 = torch.cat([d1, e4], dim=1); d1 = self.dec1(d1)
        d2 = self.up2(d1); d2 = torch.cat([d2, e3], dim=1); d2 = self.dec2(d2)
        d3 = self.up3(d2); d3 = torch.cat([d3, e2], dim=1); d3 = self.dec3(d3)
        d4 = self.up4(d3); d4 = torch.cat([d4, e1], dim=1); d4 = self.dec4(d4)
        return self.final(d4)

def apply_unet_segmentation(image):
    global _weights_warning_shown
    weights_path = os.path.join(os.path.dirname(__file__), "unet_skin_lesion.pth")

    if not os.path.exists(weights_path):
        if not _weights_warning_shown:
            print("\nU-Net weights not found → Skipping segmentation for all images")
            print(f"   Expected: {weights_path}")
            print("   Add the .pth file later → segmentation will activate automatically\n")
            _weights_warning_shown = True
        return image.copy()  # No segmentation → return original

    # If weights exist → run real U-Net
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNet().to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()

        h, w = image.shape[:2]
        img_resized = cv2.resize(image, (224, 224))
        tensor = torch.from_numpy(img_resized.transpose(2,0,1)).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            mask = torch.sigmoid(model(tensor)) > 0.5

        mask = mask[0,0].cpu().numpy()
        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.uint8) * 255

        return cv2.bitwise_and(image, image, mask=mask)

    except Exception as e:
        if not _weights_warning_shown:
            print(f"\nU-Net error: {e} → Skipping segmentation")
            _weights_warning_shown = True
        return image.copy()