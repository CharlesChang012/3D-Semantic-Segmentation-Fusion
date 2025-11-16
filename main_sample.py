import argparse
import os
import torch
import torch.nn as nn
import yaml
import numpy as np
# Import utilities
from utils.camera import ImageFeatureEncoder
from utils.lidar import LiDARFeatureEncoder
from utils.fusion_model import FeatureFusionModel
from utils.plot import plot_cloud
from utils.test import test_sample
from utils.losses import CELSLoss
from utils.dataloader import create_dataloaders

def main():

    # ==============================#
    #         Configurations        #
    # ==============================#
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Load configuration file
    with open("config/nuscenes.yaml", "r") as f:
        config = yaml.safe_load(f)

    # ==============================#
    #          Dataset Setup        #
    # ==============================#
    dataloaders = create_dataloaders(config)

    # ==============================#
    #             Model             #
    # ==============================#
    # Initialize encoders
    image_encoder = ImageFeatureEncoder(config, device=device)
    pcd_encoder = LiDARFeatureEncoder(config).to(device)

    # Initialize fusion model
    model = FeatureFusionModel(
        image_encoder=image_encoder,
        pcd_encoder=pcd_encoder,
        point_feat_dim=64,
        patch_tok_dim=384,
        mlp_hidden_dim=256,
        output_dim=config['train_params']['mlp_class'],
        device=device
    ).to(device)

    # Initialize Loss function
    criterion = CELSLoss(ignore_index=-100) # Cross-Entropy + Lovasz

    # Load best model
    best_model_path = os.path.join(config['test_params']['checkpoint_path'], "3DSSF.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # ==============================#
    #        Test One Sample        #
    # ==============================#
    test_sample_result = test_sample(
        dataloaders=dataloaders,
        image_encoder=image_encoder,
        pcd_encoder=pcd_encoder,
        model=model,
        criterion=criterion,
        device=device,
    )
 
    plot_cloud(test_sample_result['points'][0, :, :3], np.array(test_sample_result['predictions']), save_dir=config['test_params']['checkpoint_path'])


if __name__ == "__main__":
    main()