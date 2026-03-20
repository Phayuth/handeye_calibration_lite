import numpy as np
import cv2

np.set_printoptions(precision=4, suppress=True)


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
            # objPoints, imgPoints = self.board.matchImagePoints(
            #     corners,
            #     ids,
            #     None,
            #     None,
            # )
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
