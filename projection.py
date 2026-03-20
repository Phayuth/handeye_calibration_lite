import matplotlib

matplotlib.use("tkagg")  # fixed cv and plt conflict
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.transformations import plot_transform
import cv2
import yaml
from itertools import product

np.set_printoptions(precision=4, suppress=True)

with open("./calib_log/left.yaml", "r") as f:
    camparam_yaml = f.read()
camparams = yaml.safe_load(camparam_yaml)
k = camparams["camera_matrix"]["data"]
k = np.array(k).reshape(3, 3)
d = camparams["distortion_coefficients"]["data"]
d = np.array([d])
image_width = camparams["image_width"]
image_height = camparams["image_height"]
p = camparams["projection_matrix"]["data"]
p = np.array(p).reshape(3, 4)

print("Camera Matrix (k):")
print(k)
print("\nDistortion Coefficients (d):")
print(d)
print("\nProjection Matrix (p):")
print(p)

Hcamtoworld = np.array(
    [
        -0.7519,
        0.3309,
        -0.5702,
        1.0000,
        0.6577,
        0.4348,
        -0.6151,
        1.0000,
        0.0444,
        -0.8375,
        -0.5446,
        1.0000,
        0.0000,
        0.0000,
        0.0000,
        1.0000,
    ]
).reshape(4, 4)
Hworldtocam = np.linalg.inv(Hcamtoworld)
Rworldtocam = Hworldtocam[:3, :3]
tworldtocam = Hworldtocam[:3, 3].reshape(3, 1)
rworldtocamvec, _ = cv2.Rodrigues(Rworldtocam)


def project_world_points(points_world, camera_matrix, distortion_coeffs):
    points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 1, 3)
    image_points, _ = cv2.projectPoints(
        points_world, rworldtocamvec, tworldtocam, camera_matrix, distortion_coeffs
    )
    return image_points.reshape(-1, 2)


# single point
ptoworld = np.array([0.2, 0.2, 0.2])
image_point = project_world_points(ptoworld, k, d)
print(f"The 3D point in world coordinates: {ptoworld}")
print(f"The projected 2D image point: {image_point.flatten()}")
image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
tt = Hworldtocam[0:3, 3]
rr = Hworldtocam[0:3, 0:3]
cv2.drawFrameAxes(image, k, d, rr, tt, 0.5)
px = int(image_point.flatten()[0])
py = int(image_point.flatten()[1])
cv2.circle(image, (px, py), 5, (0, 0, 255), -1)
plt.imshow(image)
plt.show()

# cube of points
cube3dpoints = np.array(list(product([0.0, 0.3], repeat=3)))
projected_cube_points = project_world_points(cube3dpoints, k, d)
cube_edges = [
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),  # Bottom square
    (4, 5),
    (5, 7),
    (7, 6),
    (6, 4),  # Top square
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),  # Connecting pillars
]

for i, j in cube_edges:
    pt1 = tuple(projected_cube_points[i].astype(int))
    pt2 = tuple(projected_cube_points[j].astype(int))
    cv2.line(image, pt1, pt2, (0, 255, 255), 1)
for pt in projected_cube_points:
    cv2.circle(image, tuple(pt.astype(int)), 3, (255, 0, 255), -1)
plt.imshow(image)
plt.show()

# recover camera pose from 3D-2D correspondences using solvePnP
# objectPoints: 3D points in the world coordinate system (cube3dpoints)
# imagePoints: Corresponding 2D points in the image plane (projected_cube_points)
# cameraMatrix: Intrinsic camera matrix (k)
# distCoeffs: Distortion coefficients (d)
# ensure data types are correct for solvePnP
object_points_pnp = cube3dpoints.astype(np.float32)
image_points_pnp = projected_cube_points.astype(np.float32)
k_pnp = k.astype(np.float32)
d_pnp = d.astype(np.float32)

success, rvec_estimated, tvec_estimated = cv2.solvePnP(
    object_points_pnp, image_points_pnp, k_pnp, d_pnp
)

print("Estimated rvec (rotation vector):")
print(rvec_estimated)
print("\nEstimated tvec (translation vector):")
print(tvec_estimated)
R_estimated, _ = cv2.Rodrigues(rvec_estimated)
Hworldtocam_estimated = np.eye(4)
Hworldtocam_estimated[:3, :3] = R_estimated
Hworldtocam_estimated[:3, 3] = tvec_estimated.flatten()
print("\nEstimated H_world_to_cam (from solvePnP):")
print(Hworldtocam_estimated)
print("\nOriginal H_world_to_cam (for comparison):")
print(Hworldtocam)
