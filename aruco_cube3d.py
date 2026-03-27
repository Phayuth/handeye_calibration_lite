import cv2
import numpy as np


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

    def get_inner_2aruco_corners_size(self, new_square_size):
        markerLSRatio = self.markerLength / self.markerSeparation
        markerS = new_square_size / (2 + 2 * markerLSRatio)
        inner2aruco = (2 * markerLSRatio + 1) * markerS
        return inner2aruco

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

        # save debug
        self._cube_corners3d = cube_corners3d
        self._marker_corners3d = marker_corners3d
        self._square_corners3d = square_corners3d
        self._objPointsMarkerID = objPointsMarkerID

    def matchImagePoints(self, detectedCorners, detectedIds):
        # corners is list of (1,4,2)
        # ids is list of list of [1]
        # output shape of objPoints: (144, 1, 3), shape of imgPoints: (144, 1, 2)
        if detectedIds is None or len(detectedIds) == 0:
            return None, None

        matched_obj = []
        matched_img = []

        for det_id, corner in zip(detectedIds.flatten(), detectedCorners):
            marker_id = int(det_id)
            if marker_id in self.objPointsMarkerID:
                idx = self.objPointsMarkerID.index(marker_id)
                matched_obj.append(self.objPoints3D[idx])
                matched_img.append(corner[0])

        if not matched_obj:
            return None, None

        objPoints = np.asarray(matched_obj, dtype=np.float32).reshape(-1, 1, 3)
        imgPoints = np.asarray(matched_img, dtype=np.float32).reshape(-1, 1, 2)
        return objPoints, imgPoints

    def _plot_debug(self):
        import matplotlib.pyplot as plt
        from pytransform3d.plot_utils import make_3d_axis
        from pytransform3d.transformations import plot_transform

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(
            self._marker_corners3d[:, :, 0],
            self._marker_corners3d[:, :, 1],
            c="r",
            s=10,
        )
        for l in self._objPointsMarkerID:
            for i in l:
                for j in range(4):
                    ax.text(
                        self._marker_corners3d[i, j, 0],
                        self._marker_corners3d[i, j, 1],
                        f"{i}x{j}",
                        color="green",
                        fontsize=10,
                        ha="center",
                        va="center",
                    )
                shape = self._marker_corners3d[i]
                p = plt.Polygon(
                    shape[:, :2], closed=True, fill=None, edgecolor="b"
                )
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
        # center square
        c = plt.Polygon(
            self._square_corners3d[0, :, :2],
            closed=True,
            fill=None,
            edgecolor="g",
            hatch="///",
        )
        ax.add_patch(c)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Marker Corners (2D Projection, Image-Frame Style)")
        ax.grid()
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()  # image frame: y grows downward
        plt.show()

        axs = make_3d_axis(ax_s=0.5, unit="m", n_ticks=6)
        plot_transform(ax=axs)
        cube_corners3d = self._cube_corners3d.reshape(-1, 3)
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

        # Plot cube vertices / edges
        axs.scatter(
            cube_corners3d[:, 0],
            cube_corners3d[:, 1],
            cube_corners3d[:, 2],
            c="k",
            s=80,
            label="cube_corners3d",
        )
        for i, j in cube_edges:
            axs.plot(
                [cube_corners3d[i, 0], cube_corners3d[j, 0]],
                [cube_corners3d[i, 1], cube_corners3d[j, 1]],
                [cube_corners3d[i, 2], cube_corners3d[j, 2]],
                color="tab:blue",
                linewidth=2,
            )

        # marker corners
        axs.scatter(
            self.objPoints3D[:, :, 0].flatten(),
            self.objPoints3D[:, :, 1].flatten(),
            self.objPoints3D[:, :, 2].flatten(),
            c="m",
            s=50,
            label="objPoints",
        )
        axs.set_xticklabels([])
        axs.set_yticklabels([])
        axs.set_zticklabels([])
        axs.legend()
        axs.set_title("Marker Corners in 3D")
        plt.show()


class ARUCOCubePose:

    def __init__(self):
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.size = (6, 6)  # (cols, rows)
        self.markerLength = 0.0275
        self.markerSeparation = 0.006875
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

    def generate(self):
        cols, rows = self.size
        board_w = cols * self.markerLength + (cols - 1) * self.markerSeparation
        board_h = rows * self.markerLength + (rows - 1) * self.markerSeparation
        inner2acorner_size = self.cube.get_inner_2aruco_corners_size(0.14)
        print(f"==>> inner2acorner_size: \n{inner2acorner_size}")
        out_h = 1000
        out_w = int(
            round(out_h * (board_w / board_h))
        )  # outSize = (width, height)

        image = self.board.generateImage(
            outSize=(out_w, out_h), marginSize=20, borderBits=1
        )
        while True:
            cv2.imshow("aruco board", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

    def run(self, camera, imgraw):
        corners, ids, rej = self.detector.detectMarkers(imgraw)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(imgraw, corners, ids)  # aruco corner

            cubeobjPoints, cubeimgPoints = self.cube.matchImagePoints(corners, ids)
            if cubeobjPoints is None or len(cubeobjPoints) < 4:
                return None, None

            retval, rvc, tvc = cv2.solvePnP(
                cubeobjPoints,
                cubeimgPoints,
                camera.info["k"],
                camera.info["d"],
                None,
                None,
                False,
            )
            R, _ = cv2.Rodrigues(rvc)

            if cubeobjPoints is not None:
                cv2.drawFrameAxes(
                    imgraw,
                    camera.info["k"],
                    camera.info["d"],
                    rvc,
                    tvc,
                    1,
                    3,
                )

            return tvc, R


if __name__ == "__main__":
    from camera import Camera

    camera = Camera(4, "./calib_log/left.yaml")
    acp = ARUCOCubePose()
    acp.generate()
    while True:
        ret, imgraw = camera.read()
        if not ret:
            print("Failed to capture image")
            break
        acp.run(camera, imgraw)
        cv2.imshow("aruco board", imgraw)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
