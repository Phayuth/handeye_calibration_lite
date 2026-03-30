import pyzed.sl as sl
import cv2 as cv
import numpy as np


class ZedCamera:

    def __init__(self):
        self.zed = sl.Camera()

        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        init_params.coordinate_units = sl.UNIT.METER
        init_params.camera_fps = 60

        # Open the camera
        err = self.zed.open(init_params)
        if err > sl.ERROR_CODE.SUCCESS:
            exit(-1)

        self.zed.disable_spatial_mapping()
        self.zed.disable_body_tracking()
        self.zed.disable_positional_tracking()
        self.zed.disable_object_detection()
        self.zed.disable_recording()

        self.infoleft = {
            "height": 720,
            "width": 1280,
            "distm": "plumb_bob",
            "d": None,
            "k": None,
            "r": None,
            "p": None,
        }
        self.inforight = {
            "height": 720,
            "width": 1280,
            "distm": "plumb_bob",
            "d": None,
            "k": None,
            "r": None,
            "p": None,
        }
        self.get_camera_info()

        self.info = self.infoleft  # hack to use left only for now

    def get_camera_info(self):
        cam_info = self.zed.get_camera_information()
        Kl = (
            cam_info.camera_configuration.calibration_parameters.left_cam.fx,
            0,
            cam_info.camera_configuration.calibration_parameters.left_cam.cx,
            0,
            cam_info.camera_configuration.calibration_parameters.left_cam.fy,
            cam_info.camera_configuration.calibration_parameters.left_cam.cy,
            0,
            0,
            1,
        )
        Kr = (
            cam_info.camera_configuration.calibration_parameters.right_cam.fx,
            0,
            cam_info.camera_configuration.calibration_parameters.right_cam.cx,
            0,
            cam_info.camera_configuration.calibration_parameters.right_cam.fy,
            cam_info.camera_configuration.calibration_parameters.right_cam.cy,
            0,
            0,
            1,
        )

        self.infoleft["k"] = np.array(Kl).reshape(3, 3)
        self.inforight["k"] = np.array(Kr).reshape(3, 3)

        dl = cam_info.camera_configuration.calibration_parameters.left_cam.disto
        dr = cam_info.camera_configuration.calibration_parameters.right_cam.disto

        self.infoleft["d"] = np.array(dl)
        self.inforight["d"] = np.array(dr)

        # Hd = cam_info.camera_configuration.calibration_parameters.stereo_transform
        # Hdt = Hd.get_translation()
        # HdR = Hd.get_rotation_matrix()
        # HCamLeftToCamRight = np.eye(4)
        # HCamLeftToCamRight[:3, :3] = HdR
        # HCamLeftToCamRight[:3, 3] = Hdt

        # # RT matrix for C1 is identity.
        # # RT matrix for C2 is the R and T obtained from stereo calibration.
        # RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
        # RT2 = HCamLeftToCamRight[0:3, :]
        # # projection matrix for C1
        # # projection matrix for C2
        # self.infoleft["p"] = self.infoleft["k"] @ RT1
        # self.inforight["p"] = self.inforight["k"] @ RT2

    def read(self):
        imageleft = sl.Mat()
        imageright = sl.Mat()
        if self.zed.grab() <= sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(imageleft, sl.VIEW.LEFT)
            self.zed.retrieve_image(imageright, sl.VIEW.RIGHT)
            return imageleft, imageright
        else:
            return None, None

    def release(self):
        self.zed.close()


if __name__ == "__main__":
    zed = ZedCamera()
    while True:
        imageleft, imageright = zed.read()
        if imageleft is None or imageright is None:
            continue
        cv.imshow("LeftImage", imageleft.get_data())
        cv.imshow("Right Image", imageright.get_data())
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    zed.release()
    cv.destroyAllWindows()
