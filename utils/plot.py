import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go


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


def plot_cloud(points, labels, max_num=100000, save_dir=None):
    """
    Plot point cloud in a normal Python environment.
    """
    inds = np.arange(points.shape[0])
    inds = np.random.permutation(inds)[:max_num]

    points = points[inds]
    labels = labels[inds]

    trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            opacity=0.8,
            color=COLOR_MAP[labels].tolist(),
        )
    )

    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(aspectmode='manual',
                   aspectratio=dict(x=1, y=1, z=0.2))
    )

    fig = go.Figure(data=[trace], layout=layout)

    # This works OUTSIDE Jupyter
    save_path = os.path.join(save_dir, "segmentation_result.html")
    plotly.offline.plot(fig, filename=save_path, auto_open=True)