import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(REPO_ROOT)
from PointTransformerV3.model import PointTransformerV3

class LiDARFeatureEncoder(nn.Module):
    """
    LiDAR -> Voxelize -> PTv3 -> Point Feature
    Input:  (P x 4) tensor  [x, y, z, intensity]
    Output: (V x 64) tensor  [point-level feature map]
    """

    def __init__(self, voxel_size=0.1, pc_range=[-50, -50, -5, 50, 50, 3]):
        super().__init__()
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        # Load PTv3 backbone from the official repo
        self.ptv3 = PointTransformerV3(in_channels=4)

    def forward(self, lidar_points):
        """
        Args:
            lidar_points: (P, 4) torch.Tensor
        Returns:
            voxel_features: (V, 64)
            voxel_raw: (V, 4)
            voxel_coords: (V, 3)
        """
        voxel_raw, voxel_coords = self.voxelize_open3d(lidar_points)

        # Construct PointTransformerV3 input
        voxel_input = {
            "coord": voxel_raw[:, :3],
            "feat": voxel_raw,
            "grid_size": torch.tensor(self.voxel_size, device=lidar_points.device),
            "batch": torch.zeros(voxel_raw.shape[0], dtype=torch.long, device=lidar_points.device)
        }

        # Forward through PTv3
        voxel_output = self.ptv3(voxel_input)
        voxel_features = voxel_output.feat  # (V, C), typically last decoder layer (64-dim)

        return voxel_features, voxel_raw, voxel_coords

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
            (down_points - np.array(self.pc_range[:3])) / self.voxel_size
        ).astype(np.int32)

        # Step 5: Convert to torch
        voxel_raw = torch.from_numpy(voxel_raw).float().to(lidar_points.device)     # (V, 4) raw per-voxel inputs: x,y,z,(intensity)
        voxel_coords = torch.from_numpy(voxel_coords).int().to(lidar_points.device) # (V, 3) voxel grid coords

        return voxel_raw, voxel_coords


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
