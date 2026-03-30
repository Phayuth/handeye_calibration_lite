import json
import numpy as np
import cv2
import calibrator
import solver
import matplotlib.pyplot as plt
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.transformations import plot_transform
import yaml

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


if __name__ == "__main__":
    X = custom_solver(dataset)
    print("Custom Solver Result:\n", X)
    X_cv2 = cv2_solver(dataset)
    print("OpenCV Solver Result:\n", X_cv2)
    print("Ground Truth:\n", result_gt)
    view_handeye_result()
    view_stereo_result()
