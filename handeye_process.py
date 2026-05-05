import json
import numpy as np
import cv2
import calibrator
import solver
import matplotlib.pyplot as plt
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.transformations import plot_transform
import yaml
from itertools import product

log = json.load(open("calib_log/handeye_calibration_log.json", "r"))
dataset = log["Transform Dataset"]
result_gt = log["Result in Matrix Format"]
result_gt = np.array(result_gt).reshape(4, 4)
samples = []


def custom_solver(dataset):
    # bTe mean ee to base
    # cTo mean target to cam
    for i, data in enumerate(dataset):
        bTe = np.array(data[0]).reshape(4, 4)
        cTo = np.array(data[1]).reshape(4, 4)
        samples.append((bTe, cTo))
        print(f"Sample {i}:")
        print("bTe:\n", bTe)
        print("cTo:\n", cTo)

    # custom solver
    solver_cri = calibrator.HandEyeCalibrator(setup="Moving")
    for sample in samples:
        solver_cri.add_sample(sample[0], sample[1])
    X = solver_cri.solve(method=solver.Daniilidis1999)
    return X


def cv2_solver(dataset):
    # cv2 solver
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []
    for i, data in enumerate(dataset):
        bTe = np.array(data[0]).reshape(4, 4)
        cTo = np.array(data[1]).reshape(4, 4)

        R_gripper2base.append(bTe[:3, :3])
        t_gripper2base.append(bTe[:3, 3])

        R_target2cam.append(cTo[:3, :3])
        t_target2cam.append(cTo[:3, 3])

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=cv2.CALIB_HAND_EYE_DANIILIDIS,
    )
    Hcam2gripper = np.eye(4)
    Hcam2gripper[:3, :3] = R_cam2gripper
    Hcam2gripper[:3, 3] = t_cam2gripper.flatten()
    return Hcam2gripper


def view_handeye_result():
    Hgt = json.load(open("calib_log/handeye_calibration_log.json", "r"))[
        "Result in Matrix Format"
    ]
    H = yaml.safe_load(open("calib_log/handeye_result.yaml", "r"))[
        "Result in Matrix form (row major)"
    ]
    H = np.array(H).reshape(4, 4)

    ax = make_3d_axis(ax_s=1, unit="m", n_ticks=6)
    plot_transform(ax=ax, A2B=np.eye(4), s=0.1, name="Camera Frame")
    plot_transform(ax=ax, A2B=H, s=0.1, name="Hand-Eye Transform")
    plot_transform(ax=ax, A2B=Hgt, s=0.1, name="GT Transform")
    plt.show()


def view_stereo_result():
    data = yaml.safe_load(open("calib_log/stereo_result.yaml", "r"))
    HCamRightToBoard = data["HCamRightToBoard"]
    HCamLeftToBoard = data["HCamLeftToBoard"]
    print("HCamRightToBoard:\n", HCamRightToBoard)
    print("HCamLeftToBoard:\n", HCamLeftToBoard)

    ax = plot_transform(name="board")
    plot_transform(ax, HCamRightToBoard, name="camera_right")
    plot_transform(ax, HCamLeftToBoard, name="camera_left")
    plt.show()


def projection():
    with open("./calib_log/left.yaml", "r") as f:
        camparam_yaml = f.read()
    camp = yaml.safe_load(camparam_yaml)
    k = np.array(camp["camera_matrix"]["data"]).reshape(3, 3)
    d = np.array([camp["distortion_coefficients"]["data"]])
    image_width = camp["image_width"]
    image_height = camp["image_height"]
    p = np.array(camp["projection_matrix"]["data"]).reshape(3, 4)

    Hcamtoworld = np.array(
        [
            [-0.7519, 0.3309, -0.5702, 1.0000],
            [0.6577, 0.4348, -0.6151, 1.0000],
            [0.0444, -0.8375, -0.5446, 1.0000],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ]
    )
    Hworldtocam = np.linalg.inv(Hcamtoworld)
    Rworldtocam = Hworldtocam[:3, :3]
    tworldtocam = Hworldtocam[:3, 3].reshape(3, 1)
    rworldtocamvec, _ = cv2.Rodrigues(Rworldtocam)

    def project_world_points(points_world, camera_matrix, distortion_coeffs):
        points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 1, 3)
        image_points, _ = cv2.projectPoints(
            points_world,
            rworldtocamvec,
            tworldtocam,
            camera_matrix,
            distortion_coeffs,
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

    print("Estimated rvec (rotation vector):", rvec_estimated)
    print("\nEstimated tvec (translation vector):", tvec_estimated)
    R_estimated, _ = cv2.Rodrigues(rvec_estimated)
    Hworldtocam_estimated = np.eye(4)
    Hworldtocam_estimated[:3, :3] = R_estimated
    Hworldtocam_estimated[:3, 3] = tvec_estimated.flatten()
    print("\nEstimated H_world_to_cam (from solvePnP):", Hworldtocam_estimated)
    print("\nOriginal H_world_to_cam (for comparison):", Hworldtocam)


if __name__ == "__main__":
    # solve hand-eye calibration ---------------------
    X = custom_solver(dataset)
    print("Custom Solver Result:\n", X)
    X_cv2 = cv2_solver(dataset)
    print("OpenCV Solver Result:\n", X_cv2)
    print("Ground Truth:\n", result_gt)
    view_handeye_result()
    view_stereo_result()
    # ------------------------------------------------

    # projection   -----------------------------------
    projection()
    # ------------------------------------------------
