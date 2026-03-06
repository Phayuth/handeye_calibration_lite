import cv2
import numpy as np
import yaml
np.set_printoptions(suppress=True)

class Camera:

    def __init__(self, id, infopath) -> None:
        self.capt = cv2.VideoCapture(id)
        self.info = self.load_camera_info(infopath)

    def load_camera_info(self, yamlPath):
        try:
            with open(yamlPath, "r") as f:
                p = yaml.load(f, yaml.FullLoader)
            height = p["image_height"]
            width = p["image_width"]
            distortion_model = p["distortion_model"]
            d = p["distortion_coefficients"]["data"]
            k = p["camera_matrix"]["data"]
            r = p["rectification_matrix"]["data"]
            p = p["projection_matrix"]["data"]

        except:
            "No calibration file is provide"

        info = {
            "height": height,
            "width": width,
            "distm": distortion_model,
            "d": np.array(d),
            "k": np.array(k).reshape(3, 3),
            "r": np.array(r).reshape(3, 3),
            "p": np.array(p).reshape(3, 4),
        }

        return info

    def save_camera_info(self, yamlPath):
        data = {
            "image_width": self.info["width"],
            "image_height": self.info["height"],
            "camera_name": "narrow_stereo",
            "camera_matrix": {
                "rows": 3,
                "cols": 3,
                "data": self.info["k"].flatten().tolist(),
            },
            "distortion_model": "plumb_bob",
            "distortion_coefficients": {
                "rows": 1,
                "cols": 5,
                "data": self.info["d"].tolist(),
            },
            "rectification_matrix": {
                "rows": 3,
                "cols": 3,
                "data": self.info["r"].flatten().tolist(),
            },
            "projection_matrix": {
                "rows": 3,
                "cols": 4,
                "data": self.info["p"].flatten().tolist(),
            },
        }

        with open(yamlPath, "w") as file:
            yaml.dump(data, file)

    def read(self):
        return self.capt.read()

    def release(self):
        self.capt.release()