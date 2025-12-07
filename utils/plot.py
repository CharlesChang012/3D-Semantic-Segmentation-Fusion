import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os
# Import class names loader
from utils.dataloader import load_class_dict

def plot_training_history(train_his, val_his, save_dir=None):
    x = np.arange(len(train_his))
    plt.figure()
    plt.plot(x, torch.tensor(train_his, device='cpu'))
    plt.plot(x, torch.tensor(val_his, device='cpu'))
    plt.legend(['Training top1 accuracy', 'Validation top1 accuracy'])
    plt.xticks(x)
    plt.xlabel('Epoch')
    plt.ylabel('Top1 Accuracy')
    plt.title('3DSSF')

    if save_dir is not None:
        save_path = os.path.join(save_dir, "training_history.png")
    # Save before showing
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📁 Saved training plot to: {save_path}")

    plt.show()


# ---- Point Cloud Visualization ----
COLOR_MAP = np.array([
    '#f59664', '#f5e664', '#963c1e', '#b41e50',
    '#ff0000', '#1e1eff', '#c828ff', '#5a1e96', '#ff00ff',
    '#ff96ff', '#4b004b', '#4b00af', '#00c8ff', '#3278ff',
    '#00af00', '#003c87', '#50f096', '#96f0ff', '#0000ff'
])


def plot_comparison_cloud(config, points, pred_labels, gt_labels, save_dir=None):
    """
    Visualize prediction vs ground truth side-by-side using Plotly.
    """

    # Class names + colors
    class_dict = load_class_dict(config['dataset_params']['label_mapping'], use_16_classes=True)
    num_classes = config['train_params']['mlp_class']
    class_names = [class_dict[i] for i in range(0, num_classes + 1)]

    color_pred = COLOR_MAP[pred_labels].tolist()
    color_gt = COLOR_MAP[gt_labels].tolist()

    # ---- Predicted scatter ----
    pred_trace = go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=2, opacity=0.8, color=color_pred),
        hovertext=[class_names[int(c)] for c in pred_labels],
        hoverinfo="text",
        name="Prediction"
    )

    # ---- Ground truth scatter ----
    gt_trace = go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=2, opacity=0.6, color=color_gt),
        hovertext=[class_names[int(c)] for c in gt_labels],
        hoverinfo="text",
        name="Ground Truth"
    )

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("Ground Truth", "Prediction")
    )

    fig.add_trace(gt_trace, row=1, col=1)
    fig.add_trace(pred_trace, row=1, col=2)

    fig.update_layout(
        height=700,
        width=1400,
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    # Save
    save_path = os.path.join(save_dir, "segmentation_result.html")
    fig.write_html(save_path, auto_open=True)


def plot_iou_per_class(config, iou_per_class, save_dir=None):
    """
    Plots a histogram/bar chart of IoU per class with the provided 1-16 class colors.
    Bars are sorted from highest to lowest IoU.
    """

    # Load class names as dict {0: "name", 1:"name", ...}
    class_dict = load_class_dict(
        config['dataset_params']['label_mapping'],
        use_16_classes=True
    )

    num_classes = config['train_params']['mlp_class']  # should be 16

    # Convert to ordered list matching class IDs 1→16
    class_names = [class_dict[i] for i in range(1, num_classes + 1)]

    # Convert tensor to numpy
    iou_values = (
        iou_per_class.cpu().numpy()
        if hasattr(iou_per_class, "cpu")
        else np.array(iou_per_class)
    )

    # -------- SORT BY IoU (descending) --------
    sort_idx = np.argsort(-iou_values)               # highest → lowest
    iou_values_sorted = iou_values[sort_idx]
    class_names_sorted = [class_names[i] for i in sort_idx]

    # Reorder colors according to sorted classes
    colors_sorted = COLOR_MAP[sort_idx]

    # Numeric x positions
    x = np.arange(num_classes)

    plt.figure(figsize=(12, 5))
    plt.xlabel("Class")
    plt.ylabel("IoU")
    plt.title("Per-Class IoU")
    plt.ylim(0, 1)

    # Draw bars with sorted IoU and sorted colors
    bars = plt.bar(x, iou_values_sorted, color=colors_sorted)

    # Add IoU value above each bar
    for idx, b in enumerate(bars):
        height = b.get_height()
        plt.text(
            b.get_x() + b.get_width()/2,
            height + 0.02,
            f"{height:.3f}",
            ha='center',
            va='bottom',
            fontsize=9
        )

    # Sorted x-labels
    plt.xticks(x, class_names_sorted, rotation=45)
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "per_class_iou.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved IoU plot to: {save_path}")

    plt.show()


def plot_images_with_point_cloud(
    config,
    images,               # Tensor (B, V, C, H, W)
    points,               # (N_points, 3)
    pred_labels,          # (N_points,)
    gt_labels,            # (N_points,)
    cam_intrinsics,       # (B, V, 3, 3)
    lidar2cam_extrinsics, # (B, V, 4, 4)
    save_dir,
):
    # Load class names as dict {0: "name", 1:"name", ...}
    class_dict = load_class_dict(
        config['dataset_params']['label_mapping'],
        use_16_classes=True
    )

    num_classes = config['train_params']['mlp_class']  # should be 16
    # Convert to ordered list matching class IDs 1→16
    class_names = [class_dict[i] for i in range(1, num_classes + 1)]

    os.makedirs(save_dir, exist_ok=True)

    B, V, C, H, W = images.shape
    assert B == 1, "Batch size must be 1 for visualization"

    images_np = images[0].cpu().permute(0, 2, 3, 1).numpy()
    pts = points.cpu().numpy()
    pred = pred_labels.cpu().numpy()
    gt = gt_labels.cpu().numpy()

    # Preconvert hex to RGB
    COLOR_RGB = {
        cid: np.array(mcolors.to_rgb(hex_color))  # 0–1 float
        for cid, hex_color in enumerate(COLOR_MAP)
    }

    for cam_id in range(V):

        img = images_np[cam_id]  # (H, W, 3)
        K = cam_intrinsics[0][cam_id].cpu().numpy()
        T = lidar2cam_extrinsics[0][cam_id].cpu().numpy()

        # ---------------------------------------------
        # Transform LiDAR to camera
        # ---------------------------------------------
        pts_h = np.concatenate([pts[:, :3], np.ones((pts.shape[0], 1))], axis=1)
        pts_cam = (pts_h @ T.T)[:, :3]

        in_front = pts_cam[:, 2] > 0
        pts_cam = pts_cam[in_front]
        pred_cam = pred[in_front]
        gt_cam   = gt[in_front]

        # ---------------------------------------------
        # Project to pixels
        # ---------------------------------------------
        uv = pts_cam @ K.T
        uv[:, 0] /= uv[:, 2]
        uv[:, 1] /= uv[:, 2]

        u = uv[:, 0]
        v = uv[:, 1]

        mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u = u[mask]
        v = v[mask]
        pred_cam = pred_cam[mask]
        gt_cam = gt_cam[mask]

        # ---------------------------------------------
        # Plot side-by-side using Matplotlib
        # ---------------------------------------------
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Camera {cam_id}", fontsize=18)

        # =======================
        #   LEFT: Ground Truth
        # =======================
        ax[0].imshow(img)
        ax[0].set_title("Ground Truth", fontsize=16)
        ax[0].axis("off")
        for x, y, lab in zip(u, v, gt_cam):
            ax[0].scatter(
                x, y,
                c=[COLOR_RGB[int(lab)-1]],
                s=5
            )

        # =======================
        #   RIGHT: Prediction
        # =======================
        ax[1].imshow(img)
        ax[1].set_title("Prediction", fontsize=16)
        ax[1].axis("off")

        for x, y, lab in zip(u, v, pred_cam):
            ax[1].scatter(
                x, y,
                c=[COLOR_RGB[int(lab)-1]],
                s=5
            )

        # ---------------------------------------------
        # Legend UNDER the two images
        # ---------------------------------------------
        handles = []
        labels = []

        for cid in range(1, num_classes + 1):  # iterate through classes 1..16
            rgb = COLOR_RGB[cid-1]               # get corresponding RGB color

            handles.append(
                plt.Line2D([0], [0], marker='o', color=rgb, markersize=8, linewidth=0)
            )
            labels.append(class_names[cid-1])     # proper class name

        fig.legend(
            handles, labels,
            loc="lower center",
            ncol=min(8, len(handles)),
            bbox_to_anchor=(0.5, -0.05),
            fontsize=12
        )

        # ---------------------------------------------
        # Save
        # ---------------------------------------------
        save_path = os.path.join(save_dir, f"cam_{cam_id}_prediction.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved camera projection images to {save_dir}")

