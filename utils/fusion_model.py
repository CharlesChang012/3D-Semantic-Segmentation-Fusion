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
# 1. Projector
# -------------------------
def multi_camera_projector(lidar_voxels, calib_info_list, image_sizes):
    """
    Project LiDAR points from all 6 cameras.

    Args:
        lidar_voxels: (B, V, 3) LiDAR voxel coordinates in LiDAR frame

    Returns:
        proj_uv_all: list of 6 elements, each (B, N_valid, 2)
        depth_all:   list of 6 elements, each (B, N_valid)
    """
    self.K = [sample['cam_intrinsic'] for sample in calib_info_list]
    self.Rt = [sample['cam2lidar_extrinsic'] for sample in calib_info_list]
    self.img_h = [img_size[0] for img_size in image_sizes]
    self.img_w = [img_size[1] for img_size in image_sizes]

    B, V, _ = lidar_voxels.shape
    n_cam = self.K.shape[1]
    device = lidar_voxels.device

    # Homogeneous LiDAR points (B,V,4)
    pts_h = torch.cat(
        [lidar_voxels, torch.ones(B, V, 1, device=device)], dim=-1
    )

    proj_uv_all = []
    depth_all = []

    # Loop over cameras and build dense tensors per camera: (B, V, 2) and (B, V)
    for cam_idx in range(n_cam):
        K_cam = self.K[:, cam_idx, :, :]      # (B,3,3)
        Rt_cam = self.Rt[:, cam_idx, :, :]    # (B,4,4)

        # LiDAR → camera coordinates
        cam_pts = torch.matmul(Rt_cam, pts_h.unsqueeze(-1)).squeeze(-1)  # (B,V,4)
        xyz = cam_pts[:, :, :3]
        z = xyz[:, :, 2]  # depth in camera frame

        # Camera → image coordinates
        pix = torch.matmul(K_cam, xyz.transpose(1, 2)).transpose(1, 2)   # (B,V,3)
        u = pix[:, :, 0] / (pix[:, :, 2] + 1e-12)
        v = pix[:, :, 1] / (pix[:, :, 2] + 1e-12)

        # Validity mask (front of cam + inside image bounds)
        # Note: self.img_w / img_h may be int or list; make them tensors/broadcastable if needed
        mask = (z > 0) & (u >= 0) & (u < self.img_w) & (v >= 0) & (v < self.img_h)  #(B, V)

        # Create dense per-point uv and depth tensors with sentinel -1 for invalids
        uv = torch.stack([u, v], dim=-1)  # (B, V, 2)
        uv = torch.where(mask.unsqueeze(-1), uv, torch.full_like(uv, -1.0))
        depth = torch.where(mask, z, torch.full_like(z, -1.0))  # (B, V)

        proj_uv_all.append(uv)
        depth_all.append(depth)

    # Stack per-camera to (B, n_cam, V, 2) and (B, n_cam, V)
    proj_uv_all = torch.stack(proj_uv_all, dim=1)
    depth_all = torch.stack(depth_all, dim=1)

    return proj_uv_all, depth_all


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
# 2. Fusion Model
# -------------------------
class FeatureFusionModel(nn.Module):
    def __init__(self, image_encoder=None, pcd_encoder=None, point_feat_dim=64, patch_tok_dim=384, mlp_hidden_dim=256,
                 output_dim=16):
        super().__init__()

        # Feature encoders
        self.image_encoder = image_encoder
        self.pcd_encoder = pcd_encoder

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(point_feat_dim + patch_tok_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, output_dim)
        )

    def forward(self, patch_tokens, voxel_features, voxel_coords, image_sizes, calib_info_list):
        B, V, _ = voxel_features.shape
        _, num_views, M, dim = patch_tokens.shape

        # 1. Project 3D → 2D
        pixel_coords = multi_camera_projector(voxel_coords, calib_info_list, image_sizes)     # List of (B, V, 2) for each camera

        # 2. Scale pixel coords to resized image
        pixel_coords = scale_pixel_coords(pixel_coords, image_sizes[0], (self.image_encoder.resize_size, self.image_encoder.resize_size))

        # 3. Convert to patch grid indices
        patch_xy = (pixel_coords / float(self.image_encoder.patch_size)).long()
        grid_h = grid_w = self.image_encoder.resize_size // self.image_encoder.patch_size
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