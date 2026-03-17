import json
import numpy as np
import cv2
import calibrator
import solver

dataset = json.load(open("calib_log/handeye_calibration_log.json", "r"))
dataset = dataset["Transform Dataset"]
result_gt = json.load(open("calib_log/handeye_calibration_log.json", "r"))
result_gt = result_gt["Result in Matrix Format"]
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
    print("Hcam2gripper:\n", Hcam2gripper)
