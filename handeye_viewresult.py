import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.transformations import plot_transform
import yaml
import json

result_gt = json.load(open("calib_log/camera-robot-matrix.json", "r"))
H = yaml.safe_load(open("handeye_result.yaml", "r"))[
    "Result in Matrix form (row major)"
]
H = np.array(H).reshape(4, 4)

ax = make_3d_axis(ax_s=1, unit="m", n_ticks=6)
plot_transform(ax=ax, A2B=np.eye(4), s=0.1, name="Camera Frame")
plot_transform(ax=ax, A2B=H, s=0.1, name="Hand-Eye Transform")
plot_transform(ax=ax, A2B=result_gt, s=0.1, name="GT Transform")
plt.tight_layout()
plt.show()
