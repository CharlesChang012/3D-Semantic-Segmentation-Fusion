import torch
from torchvision import transforms

import sys
import os
sys.path.append(r"C:\Users\chels\OneDrive\Desktop\3DSSF\3D-Semantic-Segmentation-Fusion\utils")
from dataloader import create_dataloaders

# Replace v1.0-mini category.json with lidarseg category.json

config = {
    "debug": True,  
    "dataset_params": {
        "train_data_loader": {
            "data_path": r"C:\Users\chels\OneDrive\Desktop\3DSSF\data\sets\nuscenes",
            "batch_size": 2,
            "shuffle": True,
            "num_workers": 0
        },
        "label_mapping": r"C:\Users\chels\OneDrive\Desktop\3DSSF\3D-Semantic-Segmentation-Fusion\config\label_mapping\nuscenes.yaml"
    }
}

dataloaders = create_dataloaders(config)

train_loader = dataloaders['train']

for batch in train_loader:
    images, image_sizes, lidar_points_padded, labels_padded, mask, calib_info, extra = batch
    print("Images shape:", images.shape)                 # (B, 6, C, H, W)
    print("Image sizes:", image_sizes)
    print("LiDAR points shape:", lidar_points_padded.shape)  # (B, max_P, 4)
    print("Labels shape:", labels_padded.shape)            # (B, max_P)
    print("Mask shape:", mask.shape)                       # (B, max_P)
    print("Calib info length:", len(calib_info))
    break  