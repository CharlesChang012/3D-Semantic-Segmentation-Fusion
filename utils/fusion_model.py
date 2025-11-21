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
def multi_camera_projector(lidar_points, cam_intrinsics, lidar2cam_extrinsics, image_sizes):
    """
    Project batched LiDAR points into multiple camera views.

    Args:
        lidar_points: (B, V, 3) tensor of LiDAR coordinates in the LiDAR frame.
        cam_intrinsics: iterable of length B; each item has shape (N_cam, 3, 3).
        lidar2cam_extrinsics: iterable of length B; each item has shape (N_cam, 4, 4).
        image_sizes: iterable of length B with tuples (H, W) for each view (assumes same size across cameras per sample).

    Returns:
        pixel_coords: (B, N_cam, V, 2) tensor with per-view pixel coordinates (u, v).
        depth: (B, N_cam, V) tensor with per-view depths in the camera frame.
        valid_mask: (B, N_cam, V) boolean mask indicating points that project inside the image with positive depth.
    """

    device = lidar_points.device
    dtype = lidar_points.dtype

    B, V, _ = lidar_points.shape
    n_cam = cam_intrinsics[0].shape[0]

    def _to_tensor(data, expected_shape_tail):
        tensors = []
        for item in data:
            if not torch.is_tensor(item):
                tensors.append(torch.as_tensor(item, dtype=torch.float32, device=device))
            else:
                tensors.append(item.to(device=device, dtype=torch.float32))
        stacked = torch.stack(tensors, dim=0)
        if stacked.shape[1:] != expected_shape_tail:
            raise ValueError(f"Expected shape (*, {expected_shape_tail}) but got {stacked.shape}.")
        return stacked

    K = _to_tensor(cam_intrinsics, (n_cam, 3, 3))  # (B, N_cam, 3, 3)
    Rt = _to_tensor(lidar2cam_extrinsics, (n_cam, 4, 4))  # (B, N_cam, 4, 4)

    img_sizes_tensor = torch.as_tensor(image_sizes, dtype=dtype, device=device)  # (B, 2) in (H, W)
    if img_sizes_tensor.shape != (B, 2):
        raise ValueError("image_sizes must be of shape (B, 2) where each entry is (H, W).")
    img_h = img_sizes_tensor[:, 0].view(B, 1)
    img_w = img_sizes_tensor[:, 1].view(B, 1)

    pts_h = torch.cat([lidar_points, torch.ones(B, V, 1, device=device, dtype=torch.float32)], dim=-1)  # (B, V, 4)

    pixel_coords = torch.full((B, n_cam, V, 2), -1.0, dtype=dtype, device=device)
    depth = torch.zeros((B, n_cam, V), dtype=dtype, device=device)
    valid_mask = torch.zeros((B, n_cam, V), dtype=torch.bool, device=device)

    for cam_idx in range(n_cam):
        K_cam = K[:, cam_idx]
        Rt_cam = Rt[:, cam_idx] # (B, 4, 4)
        Rt_cam = Rt_cam.float()

        cam_pts = torch.matmul(Rt_cam.unsqueeze(1), pts_h.unsqueeze(-1)).squeeze(-1)    # (B, 1, 4, 4) @ (B, V, 4, 1) -> (B, V, 4, 1).squeeze(-1) -> (B, V, 4)
        xyz = cam_pts[:, :, :3]     # (B, V, 3)
        z = xyz[:, :, 2]

        pix = torch.matmul(K_cam, xyz.transpose(1, 2)).transpose(1, 2)
        denom = pix[:, :, 2].clamp(min=1e-12)
        u = pix[:, :, 0] / denom
        v = pix[:, :, 1] / denom

        valid = (z > 0) & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)

        valid_mask[:, cam_idx] = valid

        coords = torch.stack([u, v], dim=-1)
        coords = torch.where(valid.unsqueeze(-1), coords, torch.full_like(coords, -1.0))
        pixel_coords[:, cam_idx] = coords
        depth[:, cam_idx] = torch.where(valid, z, torch.full_like(z, -1.0))  # (B, V)

    return pixel_coords, depth, valid_mask


def scale_pixel_coords(pixel_coords: torch.Tensor,
                       origin_img_sizes: Sequence[Sequence[int]] | torch.Tensor,
                       new_img_size: Sequence[int] | int) -> torch.Tensor:
    """Scale pixel coordinates from their original resolution to a new resolution.

    Args:
        pixel_coords: tensor of shape (B, ..., 2) representing (u, v) coordinates.
        origin_img_sizes: iterable of (H, W) or tensor with shape (B, 2).
        new_img_size: target size. Either (H_new, W_new) tuple or single integer for square resize.
    """

    if isinstance(new_img_size, int):
        H_new = W_new = float(new_img_size)
    else:
        H_new, W_new = map(float, new_img_size)

    if torch.is_tensor(origin_img_sizes):
        origin_hw = origin_img_sizes.to(device=pixel_coords.device, dtype=pixel_coords.dtype)
        if origin_hw.dim() == 1:
            origin_hw = origin_hw.unsqueeze(0)
    else:
        origin_hw = torch.as_tensor(origin_img_sizes, dtype=pixel_coords.dtype, device=pixel_coords.device)
        if origin_hw.dim() == 1:
            origin_hw = origin_hw.unsqueeze(0)

    if origin_hw.shape[-1] != 2:
        raise ValueError("origin_img_sizes must have shape (B, 2) in (H, W) order.")

    H_orig = origin_hw[:, 0].clamp(min=1e-6)
    W_orig = origin_hw[:, 1].clamp(min=1e-6)

    scale_h = H_new / H_orig
    scale_w = W_new / W_orig
    scale = torch.stack([scale_w, scale_h], dim=-1)

    expand_dims = [1] * (pixel_coords.dim() - 2) + [2]
    scale = scale.view(scale.shape[0], *expand_dims)

    return pixel_coords * scale


# -------------------------
# 2. Fusion Model
# -------------------------
class FeatureFusionModel(nn.Module):
    def __init__(self, image_encoder=None, pcd_encoder=None, point_feat_dim=64, patch_tok_dim=384, output_dim=16, device="cuda"):
        super().__init__()

        self.device = device

        # Feature encoders
        self.image_encoder = image_encoder
        self.pcd_encoder = pcd_encoder

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(point_feat_dim + patch_tok_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, patch_tokens, voxel_features, voxel_coords, image_sizes, cam_intrinsics, lidar2cam_extrinsics):
        B, V, _ = voxel_features.shape                  # (B, V, 64)
        _, num_cams, M, dim = patch_tokens.shape        # (B, 6, M, 384)

        # 1. Project 3D → 2D
        pixel_coords, _, valid_mask = multi_camera_projector(
            voxel_coords, cam_intrinsics, lidar2cam_extrinsics, image_sizes
        )

        origin_sizes = torch.as_tensor(image_sizes, dtype=pixel_coords.dtype, device=self.device)
        pixel_coords = scale_pixel_coords(
            pixel_coords,
            origin_sizes,
            (self.image_encoder.resize_size, self.image_encoder.resize_size)
        )

        grid_size = self.image_encoder.resize_size // self.image_encoder.patch_size
        total_patches = grid_size * grid_size

        patch_xy = (pixel_coords / float(self.image_encoder.patch_size)).long()
        patch_xy[..., 0] = patch_xy[..., 0].clamp(0, grid_size - 1)
        patch_xy[..., 1] = patch_xy[..., 1].clamp(0, grid_size - 1)

        point_patch_tokens = []

        for cam in range(num_cams):
            tokens = patch_tokens[:, cam].to(self.device)               # (B, M, dim)
            idx_uv = patch_xy[:, cam].to(self.device)                   # (B, V, 2)

            flat_idx = (idx_uv[..., 1] * grid_size + idx_uv[..., 0]).clamp(0, total_patches - 1)
            gather_idx = flat_idx.unsqueeze(-1).expand(-1, -1, dim)     # (B, V, dim)
            gathered = torch.gather(tokens, dim=1, index=gather_idx)    # (B, V, dim)
            point_patch_tokens.append(gathered)

        point_patch_tokens = torch.stack(point_patch_tokens, dim=1)     # (B, num_cams, V, dim)

        mask = valid_mask.unsqueeze(-1)                                 # (B, num_cams, V, 1)
        mask = mask.to(point_patch_tokens.dtype).to(self.device)

        masked_tokens = point_patch_tokens * mask
        valid_counts = mask.sum(dim=1).clamp(min=1.0)
        # Average pooling over valid camera views
        fused_img_feat = masked_tokens.sum(dim=1) / valid_counts        # (B, V, dim)
   
        fused = torch.cat([voxel_features, fused_img_feat], dim=-1)
        voxel_scores = self.mlp(fused)  # (B, V, C)
        point_scores, _, _ = self.pcd_encoder.devoxelize(voxel_scores)
        
        return point_scores