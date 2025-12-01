import argparse
import os
import torch
import torch.nn as nn
import yaml
import numpy as np
import sys
# Import utilities
from utils.camera import ImageFeatureEncoder
from utils.lidar import LiDARFeatureEncoder
from utils.fusion_model import FeatureFusionModel
from utils.plot import plot_comparison_cloud
from utils.test import test_sample
from utils.losses import CELSLoss
from utils.dataloader import create_dataloaders
from utils.logger import Logger

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/nuscenes.yaml",
        help="Path to config YAML file"
    )
    args = parser.parse_args()

    # ==============================#
    #         Configurations        #
    # ==============================#
    # Load configuration file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    print(f"Loaded config from: {args.config}")

    # ==============================#
    #             Logger            #
    # ==============================#
    # sys.stdout = Logger(config['train_params']['checkpoint_path'], "test_sample.log")
    # sys.stderr = Logger(config['train_params']['checkpoint_path'], "test_sample.log")

    # ==============================#
    #            Set device         #
    # ==============================#
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

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
        output_dim=config['train_params']['mlp_class'],
        device=device
    ).to(device)
    
    # Load best model
    best_model_path = os.path.join(config['test_params']['checkpoint_path'], "3DSSF.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Initialize Loss function
    class_weights = torch.tensor(config['dataset_params']['class_weights'], dtype=torch.float32, device=device)
    lamda_lovasz = config['train_params']['lambda_lovasz']
    criterion = CELSLoss(weight=class_weights, ignore_index=0, lamda_lovasz=lamda_lovasz) # Cross-Entropy + Lovasz

    # ==============================#
    #        Test One Sample        #
    # ==============================#
    test_sample_result = test_sample(
        config=config,
        dataloaders=dataloaders,
        image_encoder=image_encoder,
        pcd_encoder=pcd_encoder,
        model=model,
        criterion=criterion,
        device=device,
    )

    # ==============================#
    #     Plot PCD Segmentation     #
    # ==============================#
    points, pred_labels, gt_labels = test_sample_result['points'][:, :3], np.array(test_sample_result['predictions']), np.array(test_sample_result['labels'])
    plot_comparison_cloud(config, points, pred_labels, gt_labels, save_dir=config['test_params']['checkpoint_path'])

if __name__ == "__main__":
    main()