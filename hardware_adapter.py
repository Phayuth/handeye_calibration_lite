from abc import ABC, abstractmethod
import numpy as np
import cv2


class CameraBoardProvider(ABC):
    @abstractmethod
    def initialize_hardware(self):
        pass

    @abstractmethod
    def end_hardware(self):
        pass

    @abstractmethod
    def read_frame(self):
        pass

    @abstractmethod
    def get_board_pose(self):
        pass


class RobotPoseProvider(ABC):
    @abstractmethod
    def initialize_hardware(self):
        pass

    @abstractmethod
    def end_hardware(self):
        pass

    @abstractmethod
    def get_tcp_pose(self):
        pass


class ArucoCVCameraProvider(CameraBoardProvider):

    def __init__(self, camera, board):
        self.camera = camera
        self.board = board
        self.latest_pose = None

    def initialize_hardware(self):
        pass

    def end_hardware(self):
        self.camera.release()

    def read_frame(self):
        ret, img = self.camera.read()
        if ret:
            res = self.board.run(self.camera, img)
            if res is not None:
                self.latest_pose = res
        return ret, img

    def get_board_pose(self):
        return self.latest_pose


# class ArucoZEDCameraProvider(CameraBoardProvider):

#     def __init__(self, zed, board):
#         self.zed = zed
#         self.board = board
#         self.latest_pose = None

#     def initialize_hardware(self):
#         err = self.zed.open(init_params)
#         if err > sl.ERROR_CODE.SUCCESS:
#             exit(-1)

#     def end_hardware(self):
#         self.zed.close()

#     def read_frame(self):
#         if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
#             img = sl.Mat()
#             self.zed.retrieve_image(img, sl.VIEW.LEFT)
#             img_cv = img.get_data()
#             res = self.board.run(self.zed, img_cv)
#             if res is not None:
#                 self.latest_pose = res
#             return True, img_cv
#         else:
#             return False, None

#     def get_board_pose(self):
#         return self.latest_pose


class URRobotProvider(RobotPoseProvider):

    def __init__(self, hostip):
        import rtde_control
        import rtde_receive

        self.rtde_c = rtde_control.RTDEControlInterface(hostip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(hostip)

    def initialize_hardware(self):
        self.rtde_c.teachMode()

    def end_hardware(self):
        self.rtde_c.endTeachMode()

    def get_tcp_pose(self):
        tcp = self.rtde_r.getActualTCPPose()
        Rtcp, _ = cv2.Rodrigues(np.array(tcp[3:6]))
        H = np.eye(4)
        H[:3, :3] = Rtcp
        H[:3, 3] = tcp[0:3]
        return H


if __name__ == "__main__":
    import threading
    from camera import Camera
    from aruco import ARUCOBoardPose

    def camera_loop(provider):

        global frame, running, latest_cTo_H

        while running:

            ret, img = provider.read_frame()
            pose = provider.get_board_pose()
            if pose is not None:
                latest_cTo_H = pose
            if ret:
                with lock:
                    frame = img.copy()

    def robot_loop(provider):
        global running, latest_tcp
        provider.initialize_hardware()
        try:
            while running:
                latest_tcp = provider.get_tcp_pose()
                with lock:
                    latest_bTe_H = latest_tcp
                time.sleep(0.01)  # Prevent CPU spinning
        finally:
            provider.end_hardware()

    # setup
    camera = Camera(4, "./camera_param.yaml")
    board = ARUCOBoardPose()
    camera_provider = ArucoCVCameraProvider(camera, board)
    robot_provider = URRobotProvider("192.168.0.39")
    threading.Thread(
        target=camera_loop, args=(camera_provider,), daemon=True
    ).start()
    threading.Thread(
        target=robot_loop, args=(robot_provider,), daemon=True
    ).start()
