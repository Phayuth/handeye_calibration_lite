import cv2
import threading
import numpy as np
import tkinter as tk
import yaml
from camera import Camera

lock = threading.Lock()
running = True

# global variables
frame = None
corners = None
pattern_size = (10, 7)  # chessboard pattern size (columns, rows)
samples = []
camera = Camera(4, "./camera_param.yaml")


# Camera Thread
def camera_loop():
    global frame, running, corners

    while running:
        ret, img = camera.read()
        if not ret:
            continue
        # img = cv2.flip(img, 1)
        res, found_corners = cv2.findChessboardCorners(img, pattern_size)
        cv2.drawChessboardCorners(img, pattern_size, found_corners, res)

        with lock:
            frame = img
            corners = found_corners if res else None

    camera.release()


def calib(samples):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    refined_samples = []
    for i in range(len(samples)):
        img, corners = samples[i]
        refined_corners = cv2.cornerSubPix(
            img, corners, (10, 10), (-1, -1), criteria
        )
        refined_samples.append((img, refined_corners))

    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    images, corners_list = zip(*refined_samples)
    pattern_points = [pattern_points] * len(corners_list)

    h, w = images[0].shape
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(
        pattern_points, corners_list, (w, h), None, None
    )
    return rms, camera_matrix, dist_coefs, rvecs, tvecs


# GUI
class App:

    def __init__(self, root):
        self.root = root
        self.root.title("Pinhole Camera Calibration")
        self.current_w = None
        self.current_h = None
        self.camera_matrix = None
        self.dist_coefs = None

        # -------- top frame (canvases) --------
        frame_top = tk.Frame(root)
        frame_top.pack()

        self.canvas = tk.Canvas(frame_top, width=640, height=480)
        self.canvas.pack(side=tk.LEFT)

        self.canvas2 = tk.Canvas(frame_top, width=640, height=480)
        self.canvas2.pack(side=tk.LEFT)

        self.canvas_img = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.canvas_img2 = self.canvas2.create_image(0, 0, anchor=tk.NW)

        self.photo = None
        self.photo2 = None

        # -------- bottom frame --------
        frame_bottom = tk.Frame(root)
        frame_bottom.pack(fill="x")

        # buttons
        btn_frame = tk.Frame(frame_bottom)
        btn_frame.pack()

        tk.Button(btn_frame, text="Capture Sample", command=self.capture).pack(
            side=tk.LEFT
        )
        tk.Button(btn_frame, text="Calibrate", command=self.calibrate).pack(
            side=tk.LEFT
        )
        tk.Button(
            btn_frame, text="View Undistort", command=self.view_undistort
        ).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Reset", command=self.reset).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Save", command=self.save).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Exit", command=self.quit).pack(side=tk.LEFT)

        # log text
        self.text = tk.Text(frame_bottom, height=10, width=80)
        self.text.pack()

        # save path
        path_frame = tk.Frame(frame_bottom)
        path_frame.pack()

        tk.Label(path_frame, text="Save path:").pack(side="left")

        self.save_path = tk.Entry(path_frame, width=50)
        self.save_path.insert(0, "./pinhole_result.yaml")
        self.save_path.pack(side="left")

        # sample counter
        self.status = tk.Label(frame_bottom, text="Samples: 0")
        self.status.pack()

        self.update_gui()

    def np_to_photo(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        data = f"P6 {w} {h} 255 ".encode() + img.tobytes()
        return tk.PhotoImage(data=data)

    def log(self, msg):
        self.text.insert(tk.END, msg + "\n")
        self.text.see(tk.END)

    def capture(self):
        global samples, corners, frame
        with lock:
            current_frame = frame.copy() if frame is not None else None
            current_corners = corners.copy() if corners is not None else None

        if current_frame is None:
            self.log("Error: No frame available")
            return
        if current_corners is None:
            self.log("Error: No chessboard corners detected")
            return

        samples.append(
            (cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY), current_corners)
        )
        self.status.config(text=f"Samples: {len(samples)}")
        self.log(f"Captured sample #{len(samples)}")

    def calibrate(self):
        global samples
        if len(samples) < 3:
            self.log("Error: Need at least 3 samples for calibration")
            return

        self.log("Running calibration...")
        rms, camera_matrix, dist_coefs, rvecs, tvecs = calib(samples)
        self.camera_matrix = camera_matrix
        self.dist_coefs = dist_coefs

        self.log(f"Camera Matrix:\n {camera_matrix}")
        self.log(f"Distortion Coefficients:\n {dist_coefs}")
        self.log(f"Calibration done. RMS error: {rms:.4f}")

    def view_undistort(self):
        if self.camera_matrix is None or self.dist_coefs is None:
            self.log("Error: Run calibration first")
            return

        global frame
        with lock:
            current_frame = frame.copy() if frame is not None else None

        if current_frame is None:
            self.log("Error: No frame available")
            return

        h, w = current_frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coefs, (w, h), 1, (w, h)
        )
        undistorted = cv2.undistort(
            current_frame,
            self.camera_matrix,
            self.dist_coefs,
            None,
            new_camera_matrix,
        )

        self.photo2 = self.np_to_photo(undistorted)
        self.canvas2.itemconfig(self.canvas_img2, image=self.photo2)
        self.log("Displaying undistorted image")

    def reset(self):
        global samples, corners
        samples.clear()
        corners = None
        self.status.config(text="Samples: 0")
        self.log("samples reset")

    def save(self):
        if self.camera_matrix is None or self.dist_coefs is None:
            self.log("Error: Run calibration first")
            return

        path = self.save_path.get()

        data = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coefs": self.dist_coefs.tolist(),
        }
        try:
            with open(path, "w") as f:
                yaml.dump(data, f)
            self.log(f"Saved calibration to {path}")
        except Exception as e:
            self.log(f"Error saving: {e}")

    def quit(self):
        global running
        running = False
        self.root.destroy()

    def update_gui(self):

        global frame

        with lock:
            img = None if frame is None else frame.copy()

        if img is not None:

            h, w = img.shape[:2]

            # update canvas size if resolution changed
            if w != self.current_w or h != self.current_h:
                self.current_w = w
                self.current_h = h

                self.canvas.config(width=w, height=h)
                self.canvas2.config(width=w, height=h)

            self.photo = self.np_to_photo(img)
            self.canvas.itemconfig(self.canvas_img, image=self.photo)

            self.photo2 = self.photo
            self.canvas2.itemconfig(self.canvas_img2, image=self.photo2)

        self.root.after(30, self.update_gui)


# Start
threading.Thread(target=camera_loop, daemon=True).start()
root = tk.Tk()
app = App(root)
root.mainloop()
running = False
print("release resources...")
