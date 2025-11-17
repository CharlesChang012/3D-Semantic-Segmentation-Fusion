import argparse
import os
import torch
import torch.nn as nn
import yaml
import sys
# Import utilities
from utils.camera import ImageFeatureEncoder
from utils.lidar import LiDARFeatureEncoder
from utils.fusion_model import FeatureFusionModel
from utils.train import train_model
from utils.plot import plot_training_history, plot_cloud
from utils.losses import CELSLoss
from utils.dataloader import create_dataloaders, calculate_class_weights
from utils.logger import Logger

def main():

    # ==============================#
    #         Configurations        #
    # ==============================#
    # Load configuration file
    with open("config/nuscenes.yaml", "r") as f:
        config = yaml.safe_load(f)

    # ==============================#
    #             Logger            #
    # ==============================#
    sys.stdout = Logger(config['train_params']['save_dir'], "train.log")
    sys.stderr = Logger(config['train_params']['save_dir'], "train.log")

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
        mlp_hidden_dim=256,
        output_dim=config['train_params']['mlp_class'],
        device=device
    ).to(device)

    # Load best model to continue training
    # best_model_path = os.path.join(config['train_params']['checkpoint_path'], "3DSSF.pth")
    # model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Initialize Optimizer
    if config['train_params']['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['train_params']['learning_rate'], weight_decay=config['train_params']['weight_decay'])
    elif config['train_params']['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train_params']['learning_rate'], weight_decay=config['train_params']['weight_decay'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config['train_params']['learning_rate'], momentum=config['train_params']['momentum'])

    # Initialize Loss function
    class_weights = torch.tensor(config['dataset_params']['class_weights'], dtype=torch.float32, device=device)
    criterion = CELSLoss(weight=class_weights, ignore_index=-100) # Cross-Entropy + Lovasz

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
        save_dir=config['train_params']['save_dir'],
        num_epochs=config['train_params']['max_num_epochs'],
        fusion_model_name='3DSSF'
    )

    # ==============================#
    #     Plot Training History     #
    # ==============================#
    plot_training_history(train_his, val_his, save_dir=config['train_params']['save_dir'])


if __name__ == "__main__":
    main()