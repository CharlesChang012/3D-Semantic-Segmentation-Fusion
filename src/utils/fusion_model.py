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
class MultiCameraPointProjector:
    def __init__(self, K, Rt, img_size=(900, 1600)):
        """
        Multi-camera LiDAR→image projection

        Args:
            K:  (B, 6, 3, 3)  intrinsics for 6 cameras
            Rt: (B, 6, 4, 4)  LiDAR→camera extrinsics for 6 cameras
            img_size: (H, W)  image height & width for boundary check
        """
        self.K = K
        self.Rt = Rt
        self.img_h, self.img_w = img_size

    @torch.no_grad()
    def __call__(self, lidar_points):
        """
        Project LiDAR points from all 6 cameras.

        Args:
            lidar_points: (B, V, 3) LiDAR points in LiDAR frame

        Returns:
            proj_uv_all: list of 6 elements, each (B, N_valid, 2)
            depth_all:   list of 6 elements, each (B, N_valid)
        """
        B, V, _ = lidar_points.shape
        n_cam = self.K.shape[1]
        device = lidar_points.device

        # Homogeneous LiDAR points (B,V,4)
        pts_h = torch.cat(
            [lidar_points, torch.ones(B, V, 1, device=device)], dim=-1
        )

        proj_uv_all, depth_all = [], []

        # Loop over cameras
        for cam_idx in range(n_cam):
            K_cam = self.K[:, cam_idx]      # (B,3,3)
            Rt_cam = self.Rt[:, cam_idx]    # (B,4,4)

            # LiDAR → camera coordinates
            cam_pts = torch.matmul(Rt_cam, pts_h.unsqueeze(-1)).squeeze(-1)  # (B,V,4)
            xyz = cam_pts[:, :, :3]
            z = xyz[:, :, 2]  # depth in camera frame

            # Camera → image coordinates
            pix = torch.matmul(K_cam, xyz.transpose(1, 2)).transpose(1, 2)   # (B,V,3)
            u = pix[:, :, 0] / (pix[:, :, 2] + 1e-12)
            v = pix[:, :, 1] / (pix[:, :, 2] + 1e-12)

            # Validity mask (front of cam + inside image bounds)
            mask = (z > 0) & (u >= 0) & (u < self.img_w) & (v >= 0) & (v < self.img_h)

            cam_uv_valid, cam_depth_valid = [], []
            for b in range(B):
                valid_idx = mask[b]
                u_valid = u[b, valid_idx]
                v_valid = v[b, valid_idx]
                d_valid = z[b, valid_idx]
                uv_valid = torch.stack([u_valid, v_valid], dim=-1)  # (N_valid,2)
                cam_uv_valid.append(uv_valid)
                cam_depth_valid.append(d_valid)

            proj_uv_all.append(cam_uv_valid)
            depth_all.append(cam_depth_valid)

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

        # Camera + LiDAR Extrinsics
        self.projector = MultiCameraPointProjector()

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(point_feat_dim + patch_tok_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, output_dim)
        )

    def forward(self, patch_tokens, voxel_features, voxel_coords, image_sizes, K, Rt):
        B, V, _ = voxel_features.shape
        _, num_views, M, dim = patch_tokens.shape

        # 1. Project 3D → 2D
        pixel_coords = self.projector(voxel_coords, K, Rt)     # List of (B, V, 2) for each camera

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