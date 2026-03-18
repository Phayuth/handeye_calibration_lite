import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.transformations import plot_transform

markerLength = 0.1
markerSeparation = 0.05
grid = (6, 6)  # (cols, rows)
origin_marker_id = 14


def make_3d_grid_points_zflat(grid_size, marker_length, marker_separation, z=0.0):
    cols, rows = grid_size
    pitch = marker_length + marker_separation
    xs = np.arange(cols, dtype=np.float64) * pitch
    ys = np.arange(rows, dtype=np.float64) * pitch
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    zz = np.full_like(xx, z, dtype=np.float64)
    return np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))


def make_marker_corners_from_top_left(top_left_points, marker_length):
    # Corner order per marker: top-left -> top-right -> bottom-right -> bottom-left
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [marker_length, 0.0, 0.0],
            [marker_length, marker_length, 0.0],
            [0.0, marker_length, 0.0],
        ],
        dtype=np.float64,
    )
    corners = top_left_points[:, None, :] + offsets[None, :, :]
    return corners


def make_square_size(marker_length, marker_separation):
    l = (
        marker_separation / 2
        + marker_length
        + marker_separation
        + marker_length
        + marker_separation / 2
    )
    return l


def make_square_origin(marker_separation):
    return np.array(
        [
            -marker_separation / 2,
            -marker_separation / 2,
            0.0,
        ],
        dtype=np.float64,
    )


def make_cube_from_square(square_corners3d, marker_length):
    # square_corners3d shape: (4, 3)
    # cube corners order: bottom face (0-3), top face (4-7)
    cube_corners3d = np.zeros((8, 3), dtype=np.float64)
    cube_corners3d[:4] = square_corners3d
    cube_corners3d[4:] = square_corners3d + np.array([0.0, 0.0, marker_length])
    return cube_corners3d


marker_corners3d_tl = make_3d_grid_points_zflat(
    grid, markerLength, markerSeparation
)
origin_offset = marker_corners3d_tl[origin_marker_id].copy()
marker_corners3d_tl = marker_corners3d_tl - origin_offset
marker_corners3d = make_marker_corners_from_top_left(
    marker_corners3d_tl, markerLength
)


square_size = make_square_size(markerLength, markerSeparation)
square_origin = make_square_origin(markerSeparation)
square_corners3d = make_marker_corners_from_top_left(
    square_origin[None, :], square_size
)
cube_corners3d = make_cube_from_square(square_corners3d, square_size)

# order top, left, front, right, bottom
#         0,    1,     2,     3,      4
objPointsidmarker = [
    [2, 3, 8, 9],
    [12, 13, 18, 19],
    [14, 15, 20, 21],
    [16, 17, 22, 23],
    [26, 27, 32, 33],
]


topface = marker_corners3d[objPointsidmarker[0]]
leftface = marker_corners3d[objPointsidmarker[1]]
frontface = marker_corners3d[objPointsidmarker[2]]
rightface = marker_corners3d[objPointsidmarker[3]]
bottomface = marker_corners3d[objPointsidmarker[4]]


def rotate_points(points, p1, p2, theta):
    v = p2 - p1
    v = v / np.linalg.norm(v)
    vx, vy, vz = v
    K = np.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    pts = points.reshape(-1, 3)
    pts_rot = (R @ (pts - p1).T).T + p1
    return pts_rot.reshape(points.shape)


topface_p1 = square_corners3d[0, 0]
topface_p2 = square_corners3d[0, 1]
leftface_p1 = square_corners3d[0, 3]
leftface_p2 = square_corners3d[0, 0]
rightface_p1 = square_corners3d[0, 1]
rightface_p2 = square_corners3d[0, 2]
bottomface_p1 = square_corners3d[0, 2]
bottomface_p2 = square_corners3d[0, 3]


rot = -np.deg2rad(90)
topfacerot = rotate_points(topface, topface_p1, topface_p2, rot)
leftfacerot = rotate_points(leftface, leftface_p1, leftface_p2, rot)
rightfacerot = rotate_points(rightface, rightface_p1, rightface_p2, rot)
bottomfacerot = rotate_points(bottomface, bottomface_p1, bottomface_p2, rot)


objPoints = np.empty((5 * 4, 4, 3), dtype=np.float64)
print(f"==>> objPoints.shape: \n{objPoints.shape}")
objPoints[0:4] = topfacerot
objPoints[4:8] = leftfacerot
objPoints[8:12] = frontface
objPoints[12:16] = rightfacerot
objPoints[16:20] = bottomfacerot
print(f"==>> objPoints: \n{objPoints}")


fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(marker_corners3d[:, :, 0], marker_corners3d[:, :, 1], c="r", s=10)
c = plt.Polygon(square_corners3d[0, :, :2], closed=True, fill=None, edgecolor="g")
ax.add_patch(c)
for l in objPointsidmarker:
    for i in l:
        for j in range(4):
            ax.text(
                marker_corners3d[i, j, 0],
                marker_corners3d[i, j, 1],
                f"{i}x{j}",
                color="green",
                fontsize=10,
                ha="center",
                va="center",
            )
        shape = marker_corners3d[i]
        p = plt.Polygon(shape[:, :2], closed=True, fill=None, edgecolor="b")
        ax.add_patch(p)
        mean = shape[:, :2].mean(axis=0)
        ax.text(
            mean[0],
            mean[1],
            str(i),
            color="blue",
            fontsize=8,
            ha="center",
            va="center",
        )

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Marker Corners (2D Projection, Image-Frame Style)")
ax.grid()
ax.set_aspect("equal", adjustable="box")
ax.invert_yaxis()  # image frame: y grows downward
plt.show()

axs = make_3d_axis(ax_s=0.5, unit="m", n_ticks=6)
plot_transform(ax=axs)
cube_points = cube_corners3d.reshape(-1, 3)
cube_edges = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),  # bottom square
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),  # top square
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),  # vertical edges
]

# Plot cube vertices
axs.scatter(
    cube_points[:, 0],
    cube_points[:, 1],
    cube_points[:, 2],
    c="k",
    s=80,
    label="cube_corners3d",
)

# Plot cube edges
for i, j in cube_edges:
    axs.plot(
        [cube_points[i, 0], cube_points[j, 0]],
        [cube_points[i, 1], cube_points[j, 1]],
        [cube_points[i, 2], cube_points[j, 2]],
        color="tab:blue",
        linewidth=2,
    )
axs.scatter(
    objPoints[:, :, 0].flatten(),
    objPoints[:, :, 1].flatten(),
    objPoints[:, :, 2].flatten(),
    c="m",
    s=50,
    label="objPoints",
)
axs.legend()
plt.show()
