import argparse
import os
import torch
import torch.nn as nn
# Import utilities
from utils.camera import ImageFeatureEncoder
from utils.lidar import LiDARFeatureEncoder
from utils.fusion_model import FeatureFusionModel
from utils.train import train_model, test_model
from utils.plot import plot_training_history
from utils.losses import CELSLoss

def main():

    # ==============================#
    #         Configurations        #
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
    ).to(device)

    # Initialize Optimizer
    if config['train_params']['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['train_params']['learning_rate'], weight_decay=config['train_params']['weight_decay'])
    elif config['train_params']['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train_params']['learning_rate'], weight_decay=config['train_params']['weight_decay'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config['train_params']['learning_rate'], momentum=config['train_params']['momentum'])

    # Initialize Loss function
    criterion = CELSLoss(ignore_index=-100) # Cross-Entropy + Lovasz

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
    plot_training_history(train_his, val_his)

    # ==============================#
    #          Testing Loop         #
    # ==============================#
    # Load best model
    best_model_path = os.path.join(config['train_params']['save_dir'], '3DSSF.pth')
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_result = test_model(
        dataloaders=dataloaders,
        image_encoder=image_encoder,
        pcd_encoder=pcd_encoder,
        model=model,
        criterion=criterion,
        device=device
    )

if __name__ == "__main__":
    main()