import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(REPO_ROOT)
from PointTransformerV3.model import PointTransformerV3

class LiDARFeatureEncoder(nn.Module):
    """
    LiDAR -> Voxelize -> PTv3 -> Point Feature
    Input:  (P x 4) tensor  [x, y, z, intensity]
    Output: (V x 64) tensor  [point-level feature map]
    """

    def __init__(self, config):
        super().__init__()
        self.voxel_size = config['dataset_params']['lidar']['voxel_size']
        # Load PTv3 backbone from the official repo
        self.ptv3 = PointTransformerV3(in_channels=4)

    def forward(self, lidar_points):
        """
        Args:
            lidar_points: (B, P, 4) torch.Tensor
        Returns:
            voxel_features_torch: (B, V_max, 64)
            voxel_raw_torch: (B, V_max, 4)
            voxel_coords_torch: (B, V_max, 3)
            voxel_mask: (B, V_max) bool tensor, True for valid voxels
        """
        self.lidar_points_raw = lidar_points
        B = self.lidar_points_raw.shape[0]
        voxel_features_list = []
        voxel_raw_list = []
        voxel_coords_list = []
        voxel_lengths = []

        # --- Voxelize per sample ---
        for i in range(B):
            voxel_raw, voxel_coords = self.voxelize_open3d(self.lidar_points_raw[i])

            voxel_input = {
                "coord": voxel_raw[:, :3],
                "feat": voxel_raw,
                "grid_size": torch.tensor(self.voxel_size, device=self.lidar_points_raw.device),
                "batch": torch.zeros(voxel_raw.shape[0], dtype=torch.long, device=self.lidar_points_raw.device)
            }

            voxel_output = self.ptv3(voxel_input)
            voxel_features = voxel_output.feat  # (V, C)

            voxel_features_list.append(voxel_features)
            voxel_raw_list.append(voxel_raw)
            voxel_coords_list.append(voxel_coords)
            voxel_lengths.append(voxel_features.shape[0])

        # --- Pad to max voxel length ---
        V_max = max(voxel_lengths)
        C_feat = voxel_features_list[0].shape[1]

        voxel_features_torch = torch.zeros((B, V_max, C_feat), device=self.lidar_points_raw.device)
        voxel_raw_torch = torch.zeros((B, V_max, 4), device=self.lidar_points_raw.device)
        voxel_coords_torch = torch.zeros((B, V_max, 3), device=self.lidar_points_raw.device, dtype=torch.int)
        voxel_mask = torch.zeros((B, V_max), device=self.lidar_points_raw.device, dtype=torch.bool)

        for i in range(B):
            V = voxel_lengths[i]
            voxel_features_torch[i, :V] = voxel_features_list[i]
            voxel_raw_torch[i, :V] = voxel_raw_list[i]
            voxel_coords_torch[i, :V] = voxel_coords_list[i]
            voxel_mask[i, :V] = 1  # mark valid voxels

        self.voxel_raw = voxel_raw_torch  # store for devoxelization

        return voxel_features_torch, voxel_raw_torch, voxel_coords_torch, voxel_mask

    def voxelize_open3d(self, lidar_points):
        """
        Convert raw LiDAR point cloud into voxelized representation.
        """
        pts = lidar_points.detach().cpu().numpy()

        # Step 1: Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, :3])

        # Step 2: Apply voxel downsampling
        down_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        down_points = np.asarray(down_pcd.points)

        # Step 3: Map intensity via nearest neighbor
        if pts.shape[1] == 4:
            nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(pts[:, :3])
            _, idx = nbrs.kneighbors(down_points)
            intensities = pts[idx.squeeze(), 3].reshape(-1, 1)
            voxel_raw = np.hstack([down_points, intensities])
        else:
            voxel_raw = down_points

        # Step 4: Compute voxel coordinates
        voxel_coords = np.floor(
            down_points / self.voxel_size
        ).astype(np.int32)

        # Step 5: Convert to torch
        voxel_raw = torch.from_numpy(voxel_raw).float().to(lidar_points.device)     # (V, 4) raw per-voxel inputs: x,y,z,(intensity)
        voxel_coords = torch.from_numpy(voxel_coords).int().to(lidar_points.device) # (V, 3) voxel grid coords

        return voxel_raw, voxel_coords

    def devoxelize(self, voxel_scores):
        """
        Differentiable devoxelization.
        Maps voxel-level features back to point-level using nearest-neighbor
        in pure PyTorch (keeps autograd graph).
        """
        B, V_max, C = voxel_scores.shape
        device = voxel_scores.device

        point_scores_list = []
        voxel_indices_list = []
        point_lengths = []

        for b in range(B):
            lidar_pts = self.lidar_points_raw[b]      # (P,4)
            voxel_pts = self.voxel_raw[b]             # (V,4)
            P = lidar_pts.shape[0]
            V = voxel_pts.shape[0]

            pts_xyz = lidar_pts[:, :3]                # (P,3)
            voxel_xyz = voxel_pts[:, :3]              # (V,3)

            # --- pure PyTorch nearest neighbor ---
            # Compute L2 distances between all points and voxels
            # pts_xyz:   (P,3)
            # voxel_xyz: (V,3)
            # dist: (P,V)
            dists = torch.cdist(pts_xyz, voxel_xyz)   # differentiable!

            # nearest voxel index for each point
            indices = torch.argmin(dists, dim=1)      # (P,)

            # gather voxel scores → point scores
            # voxel_feat: (V,C) → (P,C)
            voxel_feat = voxel_scores[b, :V]          # do NOT detach
            point_scores = voxel_feat[indices]        # differentiable!

            point_scores_list.append(point_scores)
            voxel_indices_list.append(indices)
            point_lengths.append(P)

        # --- pad outputs ---
        P_max = max(point_lengths)
        point_scores_padded = torch.zeros((B, P_max, C), device=device)
        voxel_indices_padded = torch.zeros((B, P_max), device=device, dtype=torch.long)
        mask = torch.zeros((B, P_max), device=device, dtype=torch.bool)

        for b in range(B):
            P = point_lengths[b]
            point_scores_padded[b, :P] = point_scores_list[b]
            voxel_indices_padded[b, :P] = voxel_indices_list[b]
            mask[b, :P] = True

        return point_scores_padded, voxel_indices_padded, mask


if __name__ == "__main__":
    # Simulated LiDAR cloud
    lidar_pcd = torch.randn(120000, 4).cuda()  # (P, 4)

    # Instantiate encoder
    pcd_encoder = LiDARFeatureEncoder(voxel_size=0.1).cuda()

    # Forward
    voxel_features, voxel_raw, voxel_coords = pcd_encoder(lidar_pcd)

    print("Input:", voxel_features.shape)
    print("Voxelized:", voxel_raw.shape)
    print("Output features:", voxel_features.shape)
