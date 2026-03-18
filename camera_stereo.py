import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.transformations as pt
from aruco import ARUCOBoardPose
from camera import Camera

np.set_printoptions(suppress=True)


def Rt_to_H(R, t):
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t.flatten()
    return H


class StereoProcess:

    def __init__(self, camleftid, camrightid, leftpath, rightpath) -> None:
        self.camleft = Camera(camleftid, leftpath)
        self.camright = Camera(camrightid, rightpath)
        self.size = (self.camleft.info["width"], self.camleft.info["height"])

    def calibrate_stereo(self, left_stereo_path, right_stereo_path):
        if True:
            self.stereo_complex_baseline()
        else:
            self.stereo_simple_baseline()
        self.camleft.save_camera_info(left_stereo_path)
        self.camright.save_camera_info(right_stereo_path)

    def determine_camera_transform(self):
        aru = ARUCOBoardPose()
        self.HboardToCamRight = None
        self.HboardToCamLeft = None
        while True:
            _, imgrraw = self.camright.read()
            _, imglraw = self.camleft.read()
            if (res := aru.run(self.camright, imgrraw)) is not None:
                tvcr, Rr = res
                self.HboardToCamRight = Rt_to_H(Rr, tvcr.flatten())
            if (res := aru.run(self.camleft, imglraw)) is not None:
                tvcl, Rl = res
                self.HboardToCamLeft = Rt_to_H(Rl, tvcl.flatten())
            cv2.imshow("right", imgrraw)
            cv2.imshow("left", imglraw)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

        self.HCamRightToBoard = np.linalg.inv(self.HboardToCamRight)
        self.HCamLeftToBoard = np.linalg.inv(self.HboardToCamLeft)
        self.HCamRightToCamLeft = self.HboardToCamLeft @ self.HCamRightToBoard
        self.HCamLeftToCamRight = np.linalg.inv(self.HCamRightToCamLeft)

        # plot
        ax = pt.plot_transform(name="board")
        pt.plot_transform(ax, self.HCamRightToBoard, name="camera_right")
        pt.plot_transform(ax, self.HCamLeftToBoard, name="camera_left")
        plt.show()

    def stereo_simple_baseline(self, a=0):
        """
        Used when the camera baseline is near each other and relatively parallel view point which is more difficult to compute.
        """
        # relative camera pose
        self.determine_camera_transform()
        self.T = self.HCamLeftToCamRight[:3, 3].reshape(3, 1)
        self.R = self.HCamLeftToCamRight[:3, :3]

        # Determine recitied rotation matrix and projection matrix for each camera given known real world R and T
        (
            self.camleft.info["r"],
            self.camright.info["r"],
            self.camleft.info["p"],
            self.camright.info["p"],
            Q,
            validPixROI1,
            validPixROI2,
        ) = cv2.stereoRectify(
            self.camleft.info["k"],
            self.camleft.info["d"],
            self.camright.info["k"],
            self.camright.info["d"],
            self.size,
            self.R,
            self.T,
            None,
            None,
            None,
            None,
            None,
            alpha=a,
        )

        # Computes the undistortion and rectification transformation map
        self.leftmapx, self.leftmapy = cv2.initUndistortRectifyMap(
            self.camleft.info["k"],
            self.camleft.info["d"],
            self.camleft.info["r"],
            self.camleft.info["p"],
            self.size,
            cv2.CV_32FC1,
            None,
            None,
        )
        self.Rightmapx, self.Rightmapy = cv2.initUndistortRectifyMap(
            self.camright.info["k"],
            self.camright.info["d"],
            self.camright.info["r"],
            self.camright.info["p"],
            self.size,
            cv2.CV_32FC1,
            None,
            None,
        )

    def stereo_complex_baseline(self):
        """
        Used when the camera baseline is complex, large and with rotation
        """
        # relative camera pose
        self.determine_camera_transform()

        # RT matrix for C1 is identity.
        # RT matrix for C2 is the R and T obtained from stereo calibration.
        RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
        RT2 = self.HCamLeftToCamRight[0:3, :]
        # projection matrix for C1
        # projection matrix for C2
        self.camleft.info["p"] = self.camleft.info["k"] @ RT1
        self.camright.info["p"] = self.camright.info["k"] @ RT2

    def remap(self, src, mapx, mapy):
        return cv2.remap(src, mapx, mapy, cv2.INTER_LINEAR)

    def triangulate(self, point1, point2):
        if point1.dtype != "float64":
            point1 = point1.astype(np.float64)

        if point2.dtype != "float64":
            point2 = point2.astype(np.float64)

        point3d = cv2.triangulatePoints(
            self.camleft.info["p"],
            self.camright.info["p"],
            point1.reshape(2, -1),
            point2.reshape(2, -1),
            None,
        ).flatten()
        point3d /= point3d[-1]
        return point3d

    def undistort_image(self, imglraw, imgrraw):
        imglund = cv2.undistort(
            imglraw, self.camleft.info["k"], self.camleft.info["d"]
        )
        imgrund = cv2.undistort(
            imgrraw, self.camright.info["k"], self.camright.info["d"]
        )
        # opt_cam_mat, valid_roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, img.shape[:2][::-1], 0)
        # ud_img = cv2.undistort(img, camera_matrix, dist_coefs, None, opt_cam_mat)
        return imglund, imgrund


if __name__ == "__main__":
    camleftid = 4
    camrightid = 10
    cam_left_preconfig = "calib_log" + "/left.yaml"
    cam_right_preconfig = "calib_log" + "/right.yaml"
    cam_left_postconfig = "calib_log" + "/streo_left.yaml"
    cam_right_postconfig = "calib_log" + "/streo_right.yaml"

    stereo = StereoProcess(
        camleftid, camrightid, cam_left_preconfig, cam_right_preconfig
    )

    stereo.calibrate_stereo(cam_left_postconfig, cam_right_postconfig)

    print("Stereo calibration finished")
