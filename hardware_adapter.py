# TODO: adapt to diffrent camera/robot providers so that the method is generalized
# to be used in different scenarios without much code change.

from abc import ABC, abstractmethod


class CameraBoardProvider(ABC):

    @abstractmethod
    def read_frame(self):
        pass

    @abstractmethod
    def get_board_pose(self):
        pass


class RobotPoseProvider(ABC):

    @abstractmethod
    def get_tcp_pose(self):
        pass


class ArucoCameraProvider(CameraBoardProvider):

    def __init__(self, camera, board):
        self.camera = camera
        self.board = board
        self.latest_pose = None

    def read_frame(self):

        ret, img = self.camera.read()

        if ret:
            res = self.board.run(self.camera, img)
            if res is not None:
                self.latest_pose = res

        return ret, img

    def get_board_pose(self):
        return self.latest_pose


class URRobotProvider(RobotPoseProvider):

    def __init__(self, hostip):

        import rtde_receive

        self.rtde_r = rtde_receive.RTDEReceiveInterface(hostip)

    def get_tcp_pose(self):
        return np.array(self.rtde_r.getActualTCPPose())


def camera_loop(provider):

    global frame, latest_tvc, latest_R

    while running:

        ret, img = provider.read_frame()

        pose = provider.get_board_pose()

        if pose is not None:
            latest_tvc, latest_R = pose

        if ret:
            with lock:
                frame = img.copy()


def robot_loop(provider):

    global latest_tcp

    while running:
        latest_tcp = provider.get_tcp_pose()


camera_provider = ArucoCameraProvider(camera, board)
robot_provider = URRobotProvider("192.168.0.39")
threading.Thread(target=camera_loop, args=(camera_provider,), daemon=True).start()
threading.Thread(target=robot_loop, args=(robot_provider,), daemon=True).start()
