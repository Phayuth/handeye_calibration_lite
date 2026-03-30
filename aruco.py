import cv2
import numpy as np


class ARUCOSinglePose:

    def __init__(self) -> None:
        self.markerLength = 0.120  # m
        self.detectorParams = cv2.aruco.DetectorParameters()
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.detector = cv2.aruco.ArucoDetector(
            self.dictionary, self.detectorParams
        )

    def aruco_pixels(self, imageRectified):
        corners, ids, rej = self.detector.detectMarkers(imageRectified)
        centerx = None
        centery = None
        if not ids is None:
            cv2.aruco.drawDetectedMarkers(
                imageRectified, corners, ids
            )  # aruco corner
            for i in range(len(ids)):
                centerx = (corners[i][0][0][0] + corners[i][0][2][0]) / 2
                centery = (corners[i][0][0][1] + corners[i][0][2][1]) / 2
                cv2.circle(
                    imageRectified,
                    (int(centerx), int(centery)),
                    15,
                    (200, 2, 1),
                    3,
                )

        return (centerx, centery)


class ARUCOBoardPose:

    def __init__(self) -> None:
        # detection
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.size = (5, 7)  # (cols, rows)
        self.markerLength = 0.0275
        self.markerSeparation = 0.006875
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
        # OpenCV drawing APIs require a writable, contiguous Mat-compatible buffer.
        if imgraw is None:
            return None
        if imgraw.ndim == 3 and imgraw.shape[2] == 4:
            imgraw = cv2.cvtColor(imgraw, cv2.COLOR_BGRA2BGR)
        imgraw = np.ascontiguousarray(imgraw, dtype=np.uint8)
        # ------------------------------------------------------------------------
        corners, ids, rej = self.detector.detectMarkers(imgraw)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(imgraw, corners, ids)  # aruco corner

            objPoints, imgPoints = self.board.matchImagePoints(
                corners,
                ids,
                None,
                None,
            )
# ------------------------------------------------------------------------
            if objPoints is None or imgPoints is None:
                return None

            # Normalize shapes for solvePnP and validate correspondence count.
            objPoints = np.asarray(objPoints, dtype=np.float64).reshape(-1, 3)
            imgPoints = np.asarray(imgPoints, dtype=np.float64).reshape(-1, 2)
            if objPoints.shape[0] < 4 or imgPoints.shape[0] < 4:
                return None
            if objPoints.shape[0] != imgPoints.shape[0]:
                return None
# ------------------------------------------------------------------------
            retval, rvc, tvc = cv2.solvePnP(
                objPoints,
                imgPoints,
                camera.info["k"],
                camera.info["d"],
                None,
                None,
                False,
            )
# ------------------------------------------------------------------------
            if not retval:
                return None
            R, _ = cv2.Rodrigues(rvc)
# ------------------------------------------------------------------------
            if objPoints is not None:
                cv2.drawFrameAxes(
                    imgraw,
                    camera.info["k"],
                    camera.info["d"],
                    rvc,
                    tvc,
                    0.1,
                    3,
                )

            return tvc, R
        return None


if __name__ == "__main__":
    # from camera import Camera

    # camleft = Camera(4, "./calib_log/left.yaml")
    # board = ARUCOBoardPose()
    # board.generate()
    # while True:
    #     _, imglraw = camleft.read()
    #     board.run(camleft, imglraw)
    #     cv2.imshow("img raw", imglraw)
    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break
    # camleft.release()
    # cv2.destroyAllWindows()

    from camera_zed import ZedCamera

    board = ARUCOBoardPose()
    zedcam = ZedCamera()
    while True:
        imgl, imgr = zedcam.read()
        if imgl is None or imgr is None:
            continue
        imgll = imgl.get_data()
        imgrr = imgr.get_data()
        imgll = np.ascontiguousarray(imgll[:, :, :3], dtype=np.uint8)
        imgrr = np.ascontiguousarray(imgrr[:, :, :3], dtype=np.uint8)
        board.run(zedcam, imgll)
        board.run(zedcam, imgrr)
        cv2.imshow("img left", imgll)
        cv2.imshow("img right", imgrr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    zedcam.release()
    cv2.destroyAllWindows()
