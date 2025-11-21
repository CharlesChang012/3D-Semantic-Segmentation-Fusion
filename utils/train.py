import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os
import copy
# Visualization
import wandb
# Import evaluation metrics
from utils.evaluation import evaluate

def train_model(dataloaders, image_encoder, pcd_encoder, model, optimizer, criterion, device,
                save_dir=None, num_epochs=10, fusion_model_name='3DSSF', config=None):

    # Start a new wandb run to track this script.
    wandbLogger = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="pohsun-university-of-michigan",
        # Set the wandb project where this run will be logged.
        project="3DSSF",
        # Track hyperparameters and run metadata.
        config=config
    )

    val_acc_history = []
    tr_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # GLOBAL STEP COUNTER
    global_step = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}\n' + '-'*20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_points = 0

            all_preds = []
            all_labels_list = []

            dataloader_tqdm = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch}", leave=False)

            for i, batch in enumerate(dataloader_tqdm):
                images, image_sizes, lidar_points, labels, mask, cam_intrinsics, lidar2cam_extrinsics = batch
                # Move to device
                images = images.to(device)              # (B, 6, C, H, W)
                image_sizes = image_sizes.to(device)    # (B, 2)
                lidar_points = lidar_points.to(device)  # (B, max_P, 4)
                labels = labels.to(device)              # (B, max_P)
                mask = mask.to(device)                  # (B, max_P)

                B, P, _ = lidar_points.shape

                # Extract features
                with torch.set_grad_enabled(phase == 'train'):
                    # Image encoder: flatten batch and views
                    # Option 1: process all views separately
                    patch_tokens_list = []
                    global_tokens_list = []

                    for v in range(images.shape[1]):
                        feats = image_encoder(images[:, v])  # (B, M, patch_dim)
                        patch_tokens_list.append(feats["patch_features"])
                        global_tokens_list.append(feats["global_features"])

                    patch_tokens = torch.stack(patch_tokens_list, dim=1)   # (B, 6, M, patch_dim)
                    global_tokens = torch.stack(global_tokens_list, dim=1) # (B, 6, patch_dim)  

                    # Point cloud encoder
                    voxel_features, voxel_raw, voxel_coords, voxel_mask = pcd_encoder(lidar_points)  # (B,V,feat_dim), (B,V,4), (B,V,3), (B,V)

                    # Forward pass through fusion model
                    outputs = model(patch_tokens, voxel_features, voxel_coords, image_sizes, cam_intrinsics, lidar2cam_extrinsics)  # (B, V, num_classes)

                    # Compute loss and predictions using mask
                    total_loss, ce_loss, lovasz_loss, predictions, gt_labels_valid = criterion(outputs, labels, mask=mask)

                # Backpropagation and optimization
                if phase == 'train':
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                running_loss += total_loss.item()
                running_corrects += torch.sum(predictions == gt_labels_valid)
                total_points += gt_labels_valid.size(0)

                # Accumulate for full-epoch metrics (VALIDATION ONLY)
                if phase == 'val':
                    all_preds.append(predictions.detach().cpu())
                    all_labels_list.append(gt_labels_valid.detach().cpu())

                # Update tqdm bar description with current loss and accuracy
                dataloader_tqdm.set_postfix({
                    'Loss': running_loss / (i + 1),
                    'Acc': running_corrects.double() / total_points
                })

                # Log to wandb
                if phase == 'train':
                    wandb.log({
                        "train/loss": float(running_loss) / float(i + 1),
                        "train/acc": float(running_corrects) / float(total_points),
                        "step": global_step
                    })
                    global_step += 1

            epoch_acc = running_corrects.double() / total_points
            
            if phase == 'val':
                all_preds = torch.cat(all_preds, dim=0)
                all_labels = torch.cat(all_labels_list, dim=0)

                num_classes = outputs.shape[-1]

                # Evaluate metrics
                evaluation_metrics = evaluate(all_preds, all_labels, num_classes, running_loss, running_corrects.double(), total_points, i + 1)
                epoch_acc = evaluation_metrics['overall_acc']

                # Save best model weights
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(best_model_wts, os.path.join(save_dir, fusion_model_name + '.pth'))

                # Log to wandb
                wandb.log({
                    "val/loss": evaluation_metrics["loss"],
                    "val/acc": evaluation_metrics["overall_acc"],
                    "val/mean_IoU": evaluation_metrics["mean_iou"],
                    "val/mean_per_class_acc": evaluation_metrics["mean_per_class_acc"],
                    "val/precision": evaluation_metrics["precision"],
                    "val/recall": evaluation_metrics["recall"],
                    "val/f1": evaluation_metrics["f1"],
                    "epoch": epoch,
                })

            # Record accuracy history
            if phase == 'train':
                tr_acc_history.append(epoch_acc)
            else:
                val_acc_history.append(epoch_acc)

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return tr_acc_history, val_acc_history