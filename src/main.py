import argparse
import os
import torch
import torch.nn as nn
# Import utilities
from utils.camera import ImageFeatureEncoder
from utils.lidar import LiDARFeatureEncoder
from utils.fusion_model import FeatureFusionModel, get_fusion_dataloaders, train_model
from utils.plot import plot_training_history

def main():

    # ==============================#
    #         Configurations        #
    # ==============================#
    num_classes = 16
    origin_img_size = (600, 900)
    IMAGE_ENCODER = 'dinov3'

    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 2
    VOXEL_SIZE = 0.1

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # ==============================#
    #          Dataset Setup        #
    # ==============================#
    # Replace this with your actual data split dictionaries
    # Each sample: {'images': [...], 'lidar_points': ..., 'labels': ...}
    data_splits = {
        'train': train_data_list,
        'val': val_data_list
    }

    image_encoder = ImageFeatureEncoder(model_name=IMAGE_ENCODER, device=device)
    pcd_encoder = LiDARFeatureEncoder(voxel_size=VOXEL_SIZE).to(device)

    dataloaders = get_fusion_dataloaders(
        data_splits,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # ==============================#
    #             Model             #
    # ==============================#
    model = FeatureFusionModel(
        image_encoder=image_encoder,
        pcd_encoder=pcd_encoder,
        point_feat_dim=64,
        patch_tok_dim=384,
        mlp_hidden_dim=256,
        output_dim=num_classes,
        origin_img_size=origin_img_size
    ).to(device)

    # Assign camera parameters
    # Replace these with your actual intrinsic/extrinsic matrices
    model.K = intrinsic.to(device)    # torch.tensor(3,3)
    model.Rt = extrinsic.to(device)   # torch.tensor(3,4)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # ==============================#
    #          Training Loop        #
    # ==============================#
    train_his, val_his = train_model(
        dataloaders=dataloaders,
        image_encoder=image_encoder,
        pcd_encoder=pcd_encoder,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir=None,
        num_epochs=NUM_EPOCHS,
        fusion_model_name='3DSSF'
    )

    # ==============================#
    #     Plot Training History     #
    # ==============================#
    plot_training_history(train_his, val_his)


if __name__ == "__main__":
    main()