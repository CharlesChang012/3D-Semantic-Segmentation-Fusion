import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import os
# Import class names loader
from utils.dataloader import load_class_names

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
    '#ffffff', '#f59664', '#f5e664', '#963c1e', '#b41e50',
    '#ff0000', '#1e1eff', '#c828ff', '#5a1e96', '#ff00ff',
    '#ff96ff', '#4b004b', '#4b00af', '#00c8ff', '#3278ff',
    '#00af00', '#003c87', '#50f096', '#96f0ff', '#0000ff'
])


def plot_cloud(config, points, labels, max_num=100000, save_dir=None):
    """
    Plot point cloud in a normal Python environment with a
    categorical colorbar showing class-label mapping.
    """
    CLASS_NAMES = load_class_names(config['dataset_params']['label_mapping'], use_16_classes=True)

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
        hovertext=[CLASS_NAMES[int(c)] for c in labels],
        hoverinfo="text"
    )

    # --- Create a dummy scatter to generate a categorical colorbar ---
    colorbar_trace = go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(
            colorscale=[[i / (len(CLASS_NAMES)-1), COLOR_MAP[i]] for i in CLASS_NAMES],
            showscale=True,
            cmin=0,
            cmax=len(CLASS_NAMES)-1,
            colorbar=dict(
                title="Classes",
                tickvals=list(CLASS_NAMES.keys()),
                ticktext=[CLASS_NAMES[k] for k in CLASS_NAMES],
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