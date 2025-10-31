import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import math
from typing import List, Dict, Any, Sequence, Tuple
import copy
import os

# -------------------------
# 2. Projector
# -------------------------
class PointToPixelProjector:
    def __call__(self, points, K, Rt):
        B, V, _ = points.shape
        points_h = torch.cat([points, torch.ones(B, V, 1, device=points.device)], dim=-1)  # (B, V, 4)
        cam_pts = torch.matmul(Rt, points_h.unsqueeze(-1)).squeeze(-1)                    # (B, V, 3)
        pix = torch.matmul(K, cam_pts.unsqueeze(-1)).squeeze(-1)                          # (B, V, 3)
        u = pix[:, :, 0] / (pix[:, :, 2] + 1e-6)
        v = pix[:, :, 1] / (pix[:, :, 2] + 1e-6)
        return torch.stack([u, v], dim=-1)  # (B, V, 2)


def scale_pixel_coords(pixel_coords: torch.Tensor,
                       origin_img_size: Sequence[int],
                       new_img_size: Sequence[int]) -> torch.Tensor:
    W_orig, H_orig = origin_img_size
    W_new, H_new = new_img_size
    scale = torch.as_tensor([W_new / W_orig, H_new / H_orig],
                            dtype=pixel_coords.dtype,
                            device=pixel_coords.device)
    view_shape = [1] * (pixel_coords.dim() - 1) + [2]
    scale = scale.view(*view_shape)
    return pixel_coords * scale


# -------------------------
# 3. Fusion Model
# -------------------------
class FeatureFusionModel(nn.Module):
    def __init__(self, image_encoder=None, pcd_encoder=None,point_feat_dim=64, patch_tok_dim=384, mlp_hidden_dim=256,
                 output_dim=16, origin_img_size=(600, 900)):
        super().__init__()

        self.origin_img_size = origin_img_size

        # PCD projector
        self.projector = PointToPixelProjector()

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(point_feat_dim + patch_tok_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, output_dim)
        )

    def forward(self, patch_tokens, voxel_features, voxel_coords, K, Rt):
        B, V, _ = voxel_features.shape
        _, num_views, M, dim = patch_tokens.shape

        # 1. Project 3D → 2D
        pixel_coords = self.projector(voxel_coords, K, Rt)

        # 2. Scale pixel coords to resized image
        pixel_coords = scale_pixel_coords(pixel_coords, self.origin_img_size, (self.resize_size, self.resize_size))

        # 3. Convert to patch grid indices
        patch_xy = (pixel_coords / float(image_encoder.patch_size)).long()
        grid_h = grid_w = image_encoder.resize_size // image_encoder.patch_size
        patch_xy[..., 0] = patch_xy[..., 0].clamp(0, grid_w - 1)
        patch_xy[..., 1] = patch_xy[..., 1].clamp(0, grid_h - 1)

        # 4. Extract per-point patch tokens
        all_views = []
        for view in range(num_views):
            tokens = patch_tokens[:, view]  # (B, M, 384)
            flat_idx = patch_xy[:, :, 0] * grid_w + patch_xy[:, :, 1]  # (B, V)
            tokens_view = torch.stack([tokens[b, flat_idx[b]].clone() for b in range(B)], dim=0)  # (B, V, 384)
            all_views.append(tokens_view)
        point_patch_tokens = torch.stack(all_views, dim=2)  # (B, V, num_views, 384)

        # 5. Average over views
        fused_img_feat = point_patch_tokens.mean(dim=2)  # (B, V, 384)

        # 6. Fuse 2D + 3D features
        fused = torch.cat([voxel_features, fused_img_feat], dim=-1)  # (B, V, 64+384)
        return self.mlp(fused)  # (B, V, output_dim)


# -------------------------
# 4. Training Loop
# -------------------------
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

            for images, lidar_points, labels, mask in dataloader:
                # Move to device
                images = images.to(device)           # (B, 6, C, H, W)
                lidar_points = lidar_points.to(device)  # (B, max_V, 3)
                labels = labels.to(device)           # (B, max_V)
                mask = mask.to(device)               # (B, max_V)

                B, V, _ = lidar_points.shape

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
                    outputs = model(patch_tokens, voxel_features, voxel_coords, model.K, model.Rt)  # (B, V, num_classes)

                    # Compute loss ignoring padded points
                    outputs_flat = outputs.view(-1, outputs.shape[-1])  # (B*V, num_classes)
                    labels_flat = labels.view(-1)                        # (B*V,)
                    mask_flat = mask.view(-1)                            # (B*V,)
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

