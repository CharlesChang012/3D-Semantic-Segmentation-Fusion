import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os
import copy
# Import evaluation metrics
from utils.evaluation import compute_confusion_matrix, compute_iou, per_class_accuracy, overall_accuracy, precision_recall_f1

@torch.no_grad()
def test_model(dataloaders, image_encoder, pcd_encoder, model, criterion, device):
    """
    Evaluate the model on the test set.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_points = 0

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
        total_loss_batch, ce_loss, lovasz_loss, predictions, gt_label_mask = criterion(outputs, labels, mask=mask)

        # Accumulate stats
        total_loss += total_loss_batch.item()

        # Remove padded points for evaluation
        valid_pred = predictions[mask]
        valid_label = labels[mask]

        total_correct += (valid_pred == valid_label).sum().item()
        total_points += valid_label.numel()

        # Save for confusion matrix
        all_preds.append(valid_pred.cpu())
        all_labels.append(valid_label.cpu())

        # Update tqdm
        dataloader_tqdm.set_postfix({
            'Loss': total_loss / total_points,
            'Acc': total_correct / total_points
        })

    # -------------------------
    # Compute Final Metrics
    # -------------------------

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    num_classes = outputs.shape[-1]

    conf_mat = compute_confusion_matrix(all_preds, all_labels, num_classes)
    iou_per_class, miou = compute_iou(conf_mat)
    acc_per_class, mean_acc = per_class_accuracy(conf_mat)
    precision, recall, f1 = precision_recall_f1(conf_mat)

    print("\n✅ Test Results")
    print(f"Loss: {total_loss/total_points:.4f}, Overall Acc: {total_correct/total_points:.4f}")
    print(f"Mean IoU: {miou:.4f}, Mean Per-Class Acc: {mean_acc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return {
        'loss': total_loss / total_points,
        'overall_acc': total_correct / total_points,
        'mean_iou': miou.item(),
        'mean_per_class_acc': mean_acc.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }

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
    total_loss, ce_loss, lovasz_loss, preds, labels_masked = criterion(outputs, labels, mask=mask)

    # ------------------------------------------------
    # 6. Prepare arrays for metrics
    # ------------------------------------------------
    # preds and labels_masked are already masked and flattened
    all_preds = preds.cpu()
    all_labels = labels_masked.cpu()

    # ------------------------------------------------
    # 7. Compute confusion matrix & metrics
    # ------------------------------------------------
    num_classes = outputs.shape[-1]
    conf_mat = compute_confusion_matrix(all_preds, all_labels, num_classes)
    iou_per_class, miou = compute_iou(conf_mat)
    acc_per_class, mean_acc = per_class_accuracy(conf_mat)
    precision, recall, f1 = precision_recall_f1(conf_mat)

    # ------------------------------------------------
    # 8. Print results
    # ------------------------------------------------
    print("\n✅ Test Results (Single Batch)")
    print(f"Loss: {total_loss.item():.4f}")
    print(f"Overall Acc: {(all_preds == all_labels).float().mean():.4f}")
    print(f"Mean IoU: {miou:.4f}")
    print(f"Mean Per-Class Acc: {mean_acc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # ------------------------------------------------
    # 9. Return dictionary
    # ------------------------------------------------
    return {
        "loss": total_loss.item(),
        "overall_acc": (all_preds == all_labels).float().mean().item(),
        "mean_iou": miou.item(),
        "mean_per_class_acc": mean_acc.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "points": lidar_points.cpu().numpy(),
        "predictions": all_preds,
        "labels": all_labels,
    }
