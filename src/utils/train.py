import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os

def train_model(dataloader, image_encoder, pcd_encoder, model, optimizer, criterion, device,
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

            for images, image_sizes, lidar_points, labels, mask, calib_info in dataloader:
                # Move to device
                images = images.to(device)              # (B, 6, C, H, W)
                image_sizes = image_sizes.to(device)    # (B, 2)
                lidar_points = lidar_points.to(device)  # (B, max_P, 4)
                labels = labels.to(device)              # (B, max_P, 4)
                mask = mask.to(device)                  # (B, max_P)

                B, P, _ = lidar_points.shape

                # Extract features
                with torch.set_grad_enabled(phase == 'train'):
                    # Image encoder: flatten batch and views
                    # Option 1: process all views separately
                    patch_tokens_list = []
                    for v in range(images.shape[1]):
                        patch_tokens_list.append(image_encoder(images[:, v]))  # (B, M, patch_dim)
                    patch_tokens = torch.stack(patch_tokens_list, dim=1)  # (B, 6, M, patch_dim)

                    # Point cloud encoder
                    voxel_features, voxel_raw, voxel_coords = pcd_encoder(lidar_points)  # (B,V,feat_dim), (B,V,4), (B,V,3)

                    # Forward pass through fusion model
                    outputs = model(patch_tokens, voxel_features, voxel_coords, image_sizes, calib_info['cam_intrinsic'], calib_info['cam2lidar_extrinsic'])  # (B, V, num_classes)

                    # Compute loss ignoring padded points
                    outputs_flat = outputs.view(-1, outputs.shape[-1])      # (B*P, num_classes)
                    labels_flat = labels.view(-1)                           # (B*P,)
                    mask_flat = mask.view(-1)                               # (B*P,)
                    outputs_masked = outputs_flat[mask_flat]
                    labels_masked = labels_flat[mask_flat]

                    loss = criterion(outputs_masked, labels_masked)
                    _, preds = torch.max(outputs_masked, 1)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * labels_masked.size(0)
                running_corrects += torch.sum(preds == labels_masked)
                total_points += labels_masked.size(0)

            epoch_loss = running_loss / total_points
            epoch_acc = running_corrects.double() / total_points
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                if save_dir:
                    torch.save(best_model_wts, os.path.join(save_dir, fusion_model_name + '.pth'))

            if phase == 'val':
                val_acc_history.append(epoch_acc)
            else:
                tr_acc_history.append(epoch_acc)

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return tr_acc_history, val_acc_history