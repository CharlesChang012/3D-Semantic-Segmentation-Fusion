import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os
import copy
# Import evaluation metrics
from utils.evaluation import evaluate

@torch.no_grad()
def test_model(dataloaders, image_encoder, pcd_encoder, model, criterion, device):
    """
    Evaluate the model on the test set.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_points = 0
    i = 0

    all_preds = []
    all_labels = []

    print("🔍 Running test evaluation...")
    dataloader_tqdm = tqdm(dataloaders['test'], desc="Testing", leave=False)

    for images, image_sizes, lidar_points, labels, mask, cam_intrinsics, cam2lidar_extrinsics in dataloader_tqdm:

        # Move to device
        images = images.to(device)              # (B, 6, C, H, W)
        image_sizes = image_sizes.to(device)    # (B, 2)
        lidar_points = lidar_points.to(device)  # (B, P, 4)
        labels = labels.to(device)              # (B, P)
        mask = mask.to(device)                  # (B, P)

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

        # -------------------------
        # Encode LiDAR
        # -------------------------
        voxel_features, voxel_raw, voxel_coords, voxel_mask = pcd_encoder(lidar_points)

        # -------------------------
        # Forward Fusion Model
        # -------------------------
        outputs = model(
            patch_tokens,
            voxel_features,
            voxel_coords,
            image_sizes,
            cam_intrinsics,
            cam2lidar_extrinsics
        )  # (B, P, num_classes)

        # -------------------------
        # Compute Loss & Predictions
        # -------------------------
        total_loss_batch, ce_loss, lovasz_loss, predictions, gt_labels_valid = criterion(outputs, labels, mask=mask)

        # Accumulate stats
        total_loss += total_loss_batch.item()
        total_correct += torch.sum(predictions == gt_labels_valid)
        total_points += gt_labels_valid.size(0)
        i += 1

        # Save for confusion matrix
        all_preds.append(predictions.cpu())
        all_labels.append(gt_labels_valid.cpu())

        # Update tqdm
        dataloader_tqdm.set_postfix({
            'Loss': total_loss / i,
            'Acc': total_correct / total_points
        })

    # -------------------------
    # Compute Final Metrics
    # -------------------------

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    num_classes = outputs.shape[-1]

    evaluation_metrics = evaluate(all_preds, all_labels, num_classes, total_loss, total_correct, total_points, i)

    return evaluation_metrics


@torch.no_grad()
def test_sample(dataloaders, image_encoder, pcd_encoder, model, criterion, device):
    """
    Run forward pass on a single batch from the test set and compute metrics.
    """

    # ------------------------------------------------
    # 1. Load a single batch
    # ------------------------------------------------
    model.eval()
    batch = next(iter(dataloaders["train"]))    #TODO: Change back to 'test'

    (images,
     image_sizes,
     lidar_points,
     labels,
     mask,
     cam_intrinsics,
     cam2lidar_extrinsics) = batch

    # Move tensors to device
    images = images.to(device)
    image_sizes = image_sizes.to(device)
    lidar_points = lidar_points.to(device)
    labels = labels.to(device)
    mask = mask.to(device)
    cam_intrinsics = [c.to(device) for c in cam_intrinsics]
    cam2lidar_extrinsics = [c.to(device) for c in cam2lidar_extrinsics]

    B, P, _ = lidar_points.shape

    # ------------------------------------------------
    # 2. Image encoder (same as your original code)
    # ------------------------------------------------
    patch_tokens_list = []
    global_tokens_list = []

    for v in range(images.shape[1]):  # number of camera views
        feats = image_encoder(images[:, v])
        patch_tokens_list.append(feats["patch_features"])
        global_tokens_list.append(feats["global_features"])

    patch_tokens = torch.stack(patch_tokens_list, dim=1)
    global_tokens = torch.stack(global_tokens_list, dim=1)

    # ------------------------------------------------
    # 3. Point cloud encoder
    # ------------------------------------------------
    voxel_features, voxel_raw, voxel_coords, voxel_mask = pcd_encoder(lidar_points)

    # ------------------------------------------------
    # 4. Fusion model forward pass
    # ------------------------------------------------
    outputs = model(
        patch_tokens,
        voxel_features,
        voxel_coords,
        image_sizes,
        cam_intrinsics,
        cam2lidar_extrinsics
    )  # (B, P, num_classes)

    # ------------------------------------------------
    # 5. Loss + predictions using mask
    # ------------------------------------------------
    total_loss, ce_loss, lovasz_loss, predictions, gt_labels_valid = criterion(outputs, labels, mask=mask)

    # ------------------------------------------------
    # 6. Prepare arrays for metrics
    # ------------------------------------------------
    # preds and labels_masked are already masked and flattened
    all_preds = predictions.cpu()
    all_labels = gt_labels_valid.cpu()
    num_classes = outputs.shape[-1]
    total_correct = torch.sum(predictions == gt_labels_valid)
    total_points = gt_labels_valid.size(0)

    evaluation_metrics = evaluate(all_preds, all_labels, num_classes, total_loss, total_correct, total_points, 1)

    scene_data = {
        "points": lidar_points.cpu().numpy(),
        "predictions": all_preds,
        "labels": all_labels,
    }

    evaluation_results = {**evaluation_metrics, **scene_data}

    return evaluation_results