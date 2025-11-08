import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os
import copy
# Import evaluation metrics
from utils.evaluation import compute_confusion_matrix, compute_iou, per_class_accuracy, overall_accuracy, precision_recall_f1

def train_model(dataloaders, image_encoder, pcd_encoder, model, optimizer, criterion, device,
                save_dir=None, num_epochs=15, fusion_model_name='3DSSF'):

    val_acc_history = []
    tr_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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

            dataloader_tqdm = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch}", leave=False)

            for images, image_sizes, lidar_points, labels, mask, cam_intrinsics, cam2lidar_extrinsics in dataloader_tqdm:
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
                    outputs = model(patch_tokens, voxel_features, voxel_coords, image_sizes, cam_intrinsics, cam2lidar_extrinsics)  # (B, V, num_classes)

                    # Compute loss and predictions using mask
                    total_loss, ce_loss, lovasz_loss, predictions, gt_label_mask = criterion(outputs, labels, mask=mask)


                if phase == 'train':
                    optimizer.zero_grad()
                    ce_loss.backward(retain_graph=True)
                    lovasz_loss.backward()
                    optimizer.step()

                running_loss += total_loss.item()
                running_corrects += torch.sum(predictions == gt_label_mask)
                total_points += gt_label_mask.size(0)

                # Update tqdm bar description with current loss and accuracy
                dataloader_tqdm.set_postfix({
                    'Loss': running_loss / total_points,
                    'Acc': running_corrects.double() / total_points
                })

            epoch_acc = running_corrects.double() / total_points
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                if save_dir:
                    torch.save(best_model_wts, os.path.join(save_dir, fusion_model_name + '.pth'))

            if phase == 'train':
                tr_acc_history.append(epoch_acc)
            else:
                val_acc_history.append(epoch_acc)

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return tr_acc_history, val_acc_history

@torch.no_grad()
def test_model(dataloaders, image_encoder, pcd_encoder, model, criterion, device):
    """
    Evaluate the model on the test set.
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_points = 0

    print("🔍 Running test evaluation...")
    dataloader_tqdm = tqdm(dataloaders['test'], desc="Testing", leave=False)

    for images, image_sizes, lidar_points, labels, mask, calib_info in dataloader_tqdm:
        # Move to device
        images = images.to(device)
        image_sizes = image_sizes.to(device)
        lidar_points = lidar_points.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        # Extract features
        patch_tokens_list = []
        for v in range(images.shape[1]):
            patch_tokens_list.append(image_encoder(images[:, v]))
        patch_tokens = torch.stack(patch_tokens_list, dim=1)

        voxel_features, voxel_raw, voxel_coords = pcd_encoder(lidar_points)

        # Forward
        outputs = model(patch_tokens, voxel_features, voxel_coords,
                        image_sizes, calib_info['cam_intrinsic'], calib_info['cam2lidar_extrinsic'])

        # Loss
        loss, ce_loss, lovasz_loss = criterion(outputs, labels, mask=mask)

        # Accuracy
        outputs_flat = outputs.view(-1, outputs.shape[-1])
        labels_flat = labels[..., 0].view(-1)
        mask_flat = mask.view(-1)

        outputs_masked = outputs_flat[mask_flat]
        labels_masked = labels_flat[mask_flat]
        _, preds = torch.max(outputs_masked, dim=1)

        running_loss += loss.item() * labels_masked.size(0)
        running_corrects += torch.sum(preds == labels_masked)
        total_points += labels_masked.size(0)

        # Update tqdm metrics
        dataloader_tqdm.set_postfix({
            'Loss': running_loss / total_points,
            'Acc': running_corrects.double() / total_points
        })

    # Evaluation Metrics
    test_loss = running_loss / total_points
    test_acc = running_corrects.double() / total_points

    confusion_matrix = compute_confusion_matrix(preds, labels_masked, num_classes=outputs.shape[-1])
    iou_per_class, miou = compute_iou(confusion_matrix)
    acc_per_class, mean_acc = per_class_accuracy(confusion_matrix)
    precision, recall, f1 = precision_recall_f1(confusion_matrix)

    print("\n✅ Test Results")
    print(f"Loss: {test_loss:.4f}, Overall Acc: {test_acc:.4f}")
    print(f"Mean IoU: {miou:.4f}, Mean Per-Class Acc: {mean_acc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    test_result = {
        'loss': test_loss,
        'overall_acc': test_acc.item(),
        'mean_iou': miou.item(),
        'mean_per_class_acc': mean_acc.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }
    return test_result