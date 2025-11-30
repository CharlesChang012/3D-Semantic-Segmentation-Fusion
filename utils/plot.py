import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
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


def plot_cloud(config, points, labels, max_num=100000, save_dir=None):
    """
    Plot point cloud in a normal Python environment with a
    categorical colorbar showing class-label mapping.
    """
    class_dict = load_class_dict(config['dataset_params']['label_mapping'], use_16_classes=True)
    num_classes = config['train_params']['mlp_class']  # should be 16
    class_names = [class_dict[i] for i in range(1, num_classes + 1)]

    # Random sampling
    inds = np.random.permutation(points.shape[0])[:max_num]
    points = points[inds]
    labels = labels[inds]

    # Main 3D scatter plot
    trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            opacity=0.8,
            color=COLOR_MAP[labels].tolist(),
        ),
        hovertext=[class_names[int(c)] for c in labels],
        hoverinfo="text"
    )

    # --- Create a dummy scatter to generate a categorical colorbar ---
    colorbar_trace = go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(
            colorscale=[[i / (len(class_names)-1), COLOR_MAP[i]] for i in class_names],
            showscale=True,
            cmin=0,
            cmax=len(class_names)-1,
            colorbar=dict(
                title="Classes",
                tickvals=list(class_names.keys()),
                ticktext=[class_names[k] for k in class_names],
                len=1.0
            ),
            color=[0]  # dummy value
        ),
        hoverinfo="none"
    )

    layout = go.Layout(
        margin=dict(l=0, r=200, b=0, t=0),   # extra right margin for colorbar
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.2),
        )
    )

    fig = go.Figure(data=[trace, colorbar_trace], layout=layout)

    # Save as standalone HTML
    save_path = os.path.join(save_dir, "segmentation_result.html")
    plotly.offline.plot(fig, filename=save_path, auto_open=True)


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
