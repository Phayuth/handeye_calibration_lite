import matplotlib

matplotlib.use("tkagg")  # fixed cv and plt conflict
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.transformations import plot_transform
import cv2
import yaml

np.set_printoptions(precision=4, suppress=True)

with open("./camera_param.yaml", "r") as f:
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


# # single point
# ptoworld = np.array([0.2, 0.2, 0.2])
# image_point = project_world_points(ptoworld, k, d)
# print(f"The 3D point in world coordinates: {ptoworld}")
# print(f"The projected 2D image point: {image_point.flatten()}")
# image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
# tt = Hworldtocam[0:3, 3]
# rr = Hworldtocam[0:3, 0:3]
# cv2.drawFrameAxes(image, k, d, rr, tt, 0.5)
# px = int(image_point.flatten()[0])
# py = int(image_point.flatten()[1])
# cv2.circle(image, (px, py), 5, (0, 0, 255), -1)
# plt.imshow(image)
# plt.show()

# # cube of points
# cube3dpoints = np.array(list(product([0.0, 0.3], repeat=3)))
# projected_cube_points = project_world_points(cube3dpoints, k, d)
# cube_edges = [
#     (0, 1),
#     (1, 3),
#     (3, 2),
#     (2, 0),  # Bottom square
#     (4, 5),
#     (5, 7),
#     (7, 6),
#     (6, 4),  # Top square
#     (0, 4),
#     (1, 5),
#     (2, 6),
#     (3, 7),  # Connecting pillars
# ]

# for i, j in cube_edges:
#     pt1 = tuple(projected_cube_points[i].astype(int))
#     pt2 = tuple(projected_cube_points[j].astype(int))
#     cv2.line(image, pt1, pt2, (0, 255, 255), 1)
# for pt in projected_cube_points:
#     cv2.circle(image, tuple(pt.astype(int)), 3, (255, 0, 255), -1)
# plt.imshow(image)
# plt.show()

# # ultra cube of points
# points_ultra = np.array(list(product([0.0, 0.1, 0.2, 0.3, 0.4], repeat=3)))
# print(f"==>> points_ultra.shape: \n{points_ultra.shape}")
# print(f"==>> points_ultra: \n{points_ultra}")
# nsamp = 50
# samples_id = np.random.choice(
#     range(points_ultra.shape[0]), size=nsamp, replace=False
# )
# points_ultra = points_ultra[samples_id]
# ax = make_3d_axis(ax_s=1, unit="m", n_ticks=6)
# plot_transform(ax=ax)
# ax.scatter(
#     points_ultra[:, 0], points_ultra[:, 1], points_ultra[:, 2], c="r", s=100
# )
# ax.set_xbound(-0.5, 2.5)
# ax.set_ybound(-0.5, 2.5)
# ax.set_zbound(-0.5, 2.5)
# plt.show()
# image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
# proj3d_points = project_world_points(points_ultra, k, d)
# for pt in proj3d_points:
#     cv2.circle(image, tuple(pt.astype(int)), 3, (255, 0, 255), -1)
# plt.imshow(image)
# plt.show()

# # recover camera pose from 3D-2D correspondences using solvePnP
# # objectPoints: 3D points in the world coordinate system (cube3dpoints)
# # imagePoints: Corresponding 2D points in the image plane (projected_cube_points)
# # cameraMatrix: Intrinsic camera matrix (k)
# # distCoeffs: Distortion coefficients (d)
# # ensure data types are correct for solvePnP
# object_points_pnp = cube3dpoints.astype(np.float32)
# image_points_pnp = projected_cube_points.astype(np.float32)
# k_pnp = k.astype(np.float32)
# d_pnp = d.astype(np.float32)

# success, rvec_estimated, tvec_estimated = cv2.solvePnP(
#     object_points_pnp, image_points_pnp, k_pnp, d_pnp
# )

# print("Estimated rvec (rotation vector):")
# print(rvec_estimated)
# print("\nEstimated tvec (translation vector):")
# print(tvec_estimated)
# R_estimated, _ = cv2.Rodrigues(rvec_estimated)
# Hworldtocam_estimated = np.eye(4)
# Hworldtocam_estimated[:3, :3] = R_estimated
# Hworldtocam_estimated[:3, 3] = tvec_estimated.flatten()
# print("\nEstimated H_world_to_cam (from solvePnP):")
# print(Hworldtocam_estimated)
# print("\nOriginal H_world_to_cam (for comparison):")
# print(Hworldtocam)


class ARUCOGridCube4x4:

    def __init__(self, markerLength, markerSeparation):
        self.markerLength = markerLength
        self.markerSeparation = markerSeparation
        self.grid = (6, 6)  # (cols, rows)
        self.origin_marker_id = 14
        self.make_object_points()

    @staticmethod
    def make_3d_grid_points_zflat(grid_size, marker_length, marker_separation):
        z = 0.0
        cols, rows = grid_size
        pitch = marker_length + marker_separation
        xs = np.arange(cols, dtype=np.float64) * pitch
        ys = np.arange(rows, dtype=np.float64) * pitch
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        zz = np.full_like(xx, z, dtype=np.float64)
        return np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    @staticmethod
    def make_marker_corners_from_top_left(top_left_points, marker_length):
        # Corner order per marker
        # top-left -> top-right -> bottom-right -> bottom-left
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

    @staticmethod
    def make_square_size(marker_length, marker_separation):
        l = (
            marker_separation / 2
            + marker_length
            + marker_separation
            + marker_length
            + marker_separation / 2
        )
        return l

    @staticmethod
    def make_square_origin(marker_separation):
        return np.array(
            [
                -marker_separation / 2,
                -marker_separation / 2,
                0.0,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def make_cube_from_square(square_corners3d, marker_length):
        # square_corners3d shape: (4, 3)
        # cube corners order: bottom face (0-3), top face (4-7)
        cube_corners3d = np.zeros((8, 3), dtype=np.float64)
        cube_corners3d[:4] = square_corners3d
        cube_corners3d[4:] = square_corners3d + np.array([0.0, 0.0, marker_length])
        return cube_corners3d

    @staticmethod
    def rotate_points(points, p1, p2, theta):
        v = p2 - p1
        v = v / np.linalg.norm(v)
        vx, vy, vz = v
        K = np.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        pts = points.reshape(-1, 3)
        pts_rot = (R @ (pts - p1).T).T + p1
        return pts_rot.reshape(points.shape)

    def make_object_points(self):
        marker_corners3d_tl = self.make_3d_grid_points_zflat(
            self.grid, self.markerLength, self.markerSeparation
        )
        origin_offset = marker_corners3d_tl[self.origin_marker_id].copy()
        marker_corners3d_tl = marker_corners3d_tl - origin_offset
        marker_corners3d = self.make_marker_corners_from_top_left(
            marker_corners3d_tl, self.markerLength
        )

        square_size = self.make_square_size(
            self.markerLength, self.markerSeparation
        )
        square_origin = self.make_square_origin(self.markerSeparation)
        square_corners3d = self.make_marker_corners_from_top_left(
            square_origin[None, :], square_size
        )
        cube_corners3d = self.make_cube_from_square(square_corners3d, square_size)

        # order top, left, front, right, bottom
        #         0,    1,     2,     3,      4
        objPointsMarkerID = [
            [2, 3, 8, 9],
            [12, 13, 18, 19],
            [14, 15, 20, 21],
            [16, 17, 22, 23],
            [26, 27, 32, 33],
        ]

        top = marker_corners3d[objPointsMarkerID[0]]
        left = marker_corners3d[objPointsMarkerID[1]]
        front = marker_corners3d[objPointsMarkerID[2]]
        right = marker_corners3d[objPointsMarkerID[3]]
        bottom = marker_corners3d[objPointsMarkerID[4]]

        top_p1 = square_corners3d[0, 0]
        top_p2 = square_corners3d[0, 1]
        left_p1 = square_corners3d[0, 3]
        left_p2 = square_corners3d[0, 0]
        right_p1 = square_corners3d[0, 1]
        right_p2 = square_corners3d[0, 2]
        bottom_p1 = square_corners3d[0, 2]
        bottom_p2 = square_corners3d[0, 3]

        rot = -np.deg2rad(90)
        toprot = self.rotate_points(top, top_p1, top_p2, rot)
        leftrot = self.rotate_points(left, left_p1, left_p2, rot)
        rightrot = self.rotate_points(right, right_p1, right_p2, rot)
        bottomrot = self.rotate_points(bottom, bottom_p1, bottom_p2, rot)

        objPoints3D = np.empty((20, 4, 3))  # 20 markers, 4 corners each, 3xyz
        objPoints3D[0:4] = toprot
        objPoints3D[4:8] = leftrot
        objPoints3D[8:12] = front
        objPoints3D[12:16] = rightrot
        objPoints3D[16:20] = bottomrot
        self.objPoints3D = objPoints3D
        self.objPointsMarkerID = np.array(objPointsMarkerID).flatten().tolist()

    def matchImagePoints(self, detectedCorners, detectedIds):
        # corners is list of (1,4,2)
        # ids is list of list of [1]
        # output shape of objPoints: (144, 1, 3), shape of imgPoints: (144, 1, 2)

        detID = np.array(detectedIds).flatten()
        matchedIDs = np.intersect1d(detID, self.objPointsMarkerID)
        numdetected = len(matchedIDs)
        objPoints = np.full((numdetected, 4, 3), np.nan)
        imgPoints = np.full((numdetected, 4, 2), np.nan)

        for id, corner in zip(detectedIds, detectedCorners):
            if id in self.objPointsMarkerID:
                idx = self.objPointsMarkerID.index(id)
                objPts = self.objPoints3D[idx]
                imgPts = corner
                objPoints[idx] = objPts
                imgPoints[idx] = imgPts

        objPoints = objPoints.reshape(-1, 1, 3)
        imgPoints = imgPoints.reshape(-1, 1, 2)
        return objPoints, imgPoints


class ARUCOCubePose:

    def __init__(self):
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.size = (6, 6)  # (cols, rows)
        self.markerLength = 0.1 / 0.1
        self.markerSeparation = 0.05 / 0.1
        self.cube = ARUCOGridCube4x4(self.markerLength, self.markerSeparation)
        self.board = cv2.aruco.GridBoard(
            self.size,
            self.markerLength,
            self.markerSeparation,
            self.dictionary,
            None,
        )
        self.detectorParams = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(
            self.dictionary, self.detectorParams
        )

    def run(self, camera, imgraw):
        corners, ids, rej = self.detector.detectMarkers(imgraw)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(imgraw, corners, ids)  # aruco corner
            objPoints, imgPoints = self.board.matchImagePoints(
                corners,
                ids,
                None,
                None,
            )
            cubeobjPoints, cubeimgPoints = self.cube.matchImagePoints(corners, ids)


acp = ARUCOCubePose()
imgraw = cv2.imread("aruco_boardsample_6x6.png")
acp.run(None, imgraw)
while True:
    cv2.imshow("aruco board", imgraw)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
