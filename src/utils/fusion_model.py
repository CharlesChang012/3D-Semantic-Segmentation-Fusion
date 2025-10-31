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
def voxel_to_pixel(voxel_coords, K, Rt):
    B, V, _ = voxel_coords.shape
    points_h = torch.cat([voxel_coords, torch.ones(B, V, 1, device=voxel_coords.device)], dim=-1)  # (B, V, 4)
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

    def forward(self, patch_tokens, voxel_features, voxel_coords, image_sizes, K, Rt):
        B, V, _ = voxel_features.shape
        _, num_views, M, dim = patch_tokens.shape

        # 1. Project 3D → 2D
        pixel_coords = voxel_to_pixel(voxel_coords, K, Rt)   # (B, V, 2)

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