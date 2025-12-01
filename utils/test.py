import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os
import copy
# Import evaluation metrics
from utils.evaluation import evaluate
from utils.plot import plot_images_with_point_cloud

@torch.no_grad()
def test_model(dataloaders, image_encoder, pcd_encoder, model, criterion, device):
    """
    Evaluate the model on the test set.
    """
    model.eval()

    total_loss = 0.0
    total_corrects = 0
    total_points = 0

    all_preds = []
    all_labels = []

    print("🔍 Running test evaluation...")
    dataloader_tqdm = tqdm(dataloaders['test'], desc="Testing", leave=False)

    for i, batch in enumerate(dataloader_tqdm):
        images, image_sizes, lidar_points, labels, mask, cam_intrinsics, lidar2cam_extrinsics = batch
        # Move to device
        images = images.to(device)              # (B, 6, C, H, W)
        image_sizes = image_sizes.to(device)    # (B, 2) (H, W)
        lidar_points = lidar_points.to(device)  # (B, max_P, 4)
        labels = labels.to(device)              # (B, max_P)
        mask = mask.to(device)                  # (B, max_P)

        B, P, _ = lidar_points.shape

        # -------------------------
        # Encode Images
        # -------------------------
        patch_tokens_list = []
        global_tokens_list = []

        for v in range(images.shape[1]):
            feats = image_encoder(images[:, v])
            patch_tokens_list.append(feats["patch_features"])
            global_tokens_list.append(feats["global_features"])

        patch_tokens = torch.stack(patch_tokens_list, dim=1)   # (B, 6, M, patch_dim)
        global_tokens = torch.stack(global_tokens_list, dim=1) # (B, 6, patch_dim)
        
        # Point cloud encoder
        voxel_features, voxel_raw, voxel_coords, voxel_mask = pcd_encoder(lidar_points)  # (B,V,feat_dim), (B,V,4), (B,V,3), (B,V)
        
        # Forward pass through fusion model
        outputs = model(patch_tokens, voxel_features, voxel_raw, voxel_coords, image_sizes, cam_intrinsics, lidar2cam_extrinsics)  # (B, V, num_classes)
        
        # Compute loss and predictions using mask
        combined_loss, ce_loss, lovasz_loss, predictions, gt_labels_valid = criterion(outputs, labels, mask=mask)

        # Accumulate stats
        total_loss += combined_loss.item()
        total_corrects += torch.sum(predictions == gt_labels_valid)
        total_points += gt_labels_valid.size(0)

        # Save for confusion matrix
        all_preds.append(predictions.detach().cpu())
        all_labels.append(gt_labels_valid.detach().cpu())

        # Update tqdm
        dataloader_tqdm.set_postfix({
            'Loss': total_loss / (i + 1),
            'Acc': total_corrects.double() / total_points
        })

    # -------------------------
    # Compute Final Metrics
    # -------------------------

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    num_classes = outputs.shape[-1]

    evaluation_metrics = evaluate(all_preds, all_labels, num_classes, total_loss, total_corrects.double(), total_points, i + 1)

    return evaluation_metrics


@torch.no_grad()
def test_sample(config, dataloaders, image_encoder, pcd_encoder, model, criterion, device):
    """
    Run forward pass on a single batch from the test set and compute metrics.
    """

    # ------------------------------------------------
    # Load a single batch
    # ------------------------------------------------
    model.eval()
    batch = next(iter(dataloaders["test"]))

    images, image_sizes, lidar_points, labels, mask, cam_intrinsics, lidar2cam_extrinsics = batch

    # Move tensors to device
    images = images.to(device)
    image_sizes = image_sizes.to(device)
    lidar_points = lidar_points.to(device)
    labels = labels.to(device)
    mask = mask.to(device)
    cam_intrinsics = [c.to(device) for c in cam_intrinsics]
    lidar2cam_extrinsics = [c.to(device) for c in lidar2cam_extrinsics]

    B, P, _ = lidar_points.shape

    # ------------------------------------------------
    # Image encoder
    # ------------------------------------------------
    patch_tokens_list = []
    global_tokens_list = []

    for v in range(images.shape[1]):  # number of camera views
        feats = image_encoder(images[:, v])
        patch_tokens_list.append(feats["patch_features"])
        global_tokens_list.append(feats["global_features"])

    patch_tokens = torch.stack(patch_tokens_list, dim=1)   # (B, 6, M, patch_dim)
    global_tokens = torch.stack(global_tokens_list, dim=1) # (B, 6, patch_dim)

    # Point cloud encoder
    voxel_features, voxel_raw, voxel_coords, voxel_mask = pcd_encoder(lidar_points)  # (B,V,feat_dim), (B,V,4), (B,V,3), (B,V)
    
    # Forward pass through fusion model
    outputs = model(patch_tokens, voxel_features, voxel_raw, voxel_coords, image_sizes, cam_intrinsics, lidar2cam_extrinsics)  # (B, V, num_classes)
    
    # Compute loss and predictions using mask
    combined_loss, ce_loss, lovasz_loss, predictions, gt_labels_valid = criterion(outputs, labels, mask=mask)

    # ------------------------------------------------
    # Prepare arrays for metrics
    # ------------------------------------------------
    # preds and labels_masked are already masked and flattened
    valid_mask = mask.bool()  # (B, P)
    valid_pts = lidar_points[0][valid_mask[0]]          # (N_valid, 4)
    all_preds = predictions.cpu()
    all_labels = gt_labels_valid.cpu()
    num_classes = outputs.shape[-1]
    total_correct = torch.sum(predictions == gt_labels_valid)
    total_points = gt_labels_valid.size(0)

    evaluation_metrics = evaluate(all_preds, all_labels, num_classes, combined_loss, total_correct.double(), total_points, 1)

    scene_data = {
        "points": valid_pts.cpu().numpy(),
        "predictions": all_preds,
        "labels": all_labels,
    }

    evaluation_results = {**evaluation_metrics, **scene_data}


    plot_images_with_point_cloud(
        config=config,
        images=images, 
        points=valid_pts, 
        pred_labels=all_preds,
        gt_labels=all_labels,
        cam_intrinsics=cam_intrinsics,
        lidar2cam_extrinsics=lidar2cam_extrinsics,
        save_dir=config['test_params']['checkpoint_path']
    )

    return evaluation_results