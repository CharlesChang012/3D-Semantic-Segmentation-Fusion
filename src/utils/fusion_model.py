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
# 1. Dataset
# -------------------------
class FusionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        return (
            sample["images"],           # (6, C, H, W)
            sample["lidar_points"],     # (P, 4)
            sample["labels"]            # (P,)  
        )

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

            for images, lidar_points, labels in dataloader:
                images = images.to(device)
                lidar_points = lidar_points.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    patch_tokens = image_encoder(images)  # (6, M, 384)
                    voxel_features, voxel_raw, voxel_coords = pcd_encoder(lidar_points) # voxel_features: (V, 64), voxel_raw: (V, 4), voxel_coords: (V, 3)

                if phase == 'train':
                    outputs_v = model(voxel_features, patch_tokens, voxel_coords, model.K, model.Rt)
                    outputs_p = v2p(outputs_v)  # TODO: define v2p()
                    _, preds = torch.max(outputs_p, 1)
                    loss = criterion(outputs_p, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs_v = model(patch_tokens, voxel_features, voxel_coords, model.K, model.Rt)
                        outputs_p = v2p(outputs_v)
                        _, preds = torch.max(outputs_p, 1)
                        loss = criterion(outputs_p, labels)

                running_loss += loss.item() * voxel_features.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
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


# -------------------------
# 5. Dataloaders
# -------------------------
def get_fusion_dataloaders(data_splits: Dict[str, List[Dict[str, Any]]],
                           batch_size: int,
                           num_workers: int = 2
                           ):

    def collate_fn(batch: List[Any]):
        batch_voxel_features = [item[0] for item in batch]
        batch_patch_tokens = [item[1] for item in batch]
        batch_voxel_coords = [item[2] for item in batch]
        batch_labels = [item[3] for item in batch]

        B = len(batch)
        num_views, M, D = batch_patch_tokens[0].shape

        patch_tokens = torch.stack(batch_patch_tokens, dim=0)

        max_V = max([vf.shape[0] for vf in batch_voxel_features])
        feat_dim = batch_voxel_features[0].shape[1]

        voxel_feats_padded = torch.zeros((B, max_V, feat_dim), dtype=torch.float32)
        voxel_coords_padded = torch.zeros((B, max_V, 3), dtype=torch.float32)
        labels_padded = torch.full((B, max_V), -100, dtype=torch.long)  # ignore_index
        mask = torch.zeros((B, max_V), dtype=torch.bool)

        for i in range(B):
            V = batch_voxel_features[i].shape[0]
            voxel_feats_padded[i, :V] = batch_voxel_features[i]
            voxel_coords_padded[i, :V] = batch_voxel_coords[i]
            labels_padded[i, :V] = batch_labels[i]
            mask[i, :V] = 1

        return voxel_feats_padded, patch_tokens, voxel_coords_padded, labels_padded, mask

    dataloaders = {}
    for split in data_splits.keys():
        ds = FusionDataset(data_splits[split])
        dl = DataLoader(ds, batch_size=batch_size, shuffle=(split=='train'), num_workers=num_workers, collate_fn=collate_fn)
        dataloaders[split] = dl

    return dataloaders
