import cv2
import threading
import tkinter as tk
import yaml
import numpy as np
from camera import Camera
from aruco import ARUCOBoardPose
from aruco_cube3d import ARUCOCubePose

lock = threading.Lock()
running = True

# global variables
frame_left = None
frame_right = None
HobjTocamleft = None
HobjTocamright = None

camera_left = Camera(4, "./calib_log/left.yaml")
# camera_right = Camera(5, "./calib_log/right.yaml")
cube = ARUCOCubePose()


# Camera Thread
def camera_loop():
    global frame_left, frame_right, running

    while running:
        ret1, img_left = camera_left.read()
        # ret, img_right = camera_right.read()
        ret2 = ret1
        img_right = img_left.copy()  # For testing with single camera
        if not ret1 and not ret2:
            continue

        HboardToCamLeft = cube.run(camera_left, img_left)
        HboardToCamRight = cube.run(camera_left, img_right)

        with lock:
            frame_left = img_left
            frame_right = img_right

    camera_left.release()
    # camera_right.release()


def calib(samples):
    print("Calibration not implemented yet")


# GUI
class App:

    def __init__(self, root):
        self.root = root
        self.root.title("Stereo Camera Calibration")
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

        tk.Button(btn_frame, text="Calibrate", command=self.calibrate).pack(
            side=tk.LEFT
        )
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
        self.save_path.insert(0, "./stereo_result.yaml")
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

    def calibrate(self):
        self.log("Running calibration...")
        rms = calib(None)
        self.log(f"Calibration done {rms}")

    def reset(self):
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
        global frame_left, frame_right

        with lock:
            img_left = None if frame_left is None else frame_left.copy()
            img_right = None if frame_right is None else frame_right.copy()

        if img_left is not None and img_right is not None:

            h, w = img_left.shape[:2]

            # update canvas size if resolution changed
            if w != self.current_w or h != self.current_h:
                self.current_w = w
                self.current_h = h

                self.canvas.config(width=w, height=h)
                self.canvas2.config(width=w, height=h)

            self.photo = self.np_to_photo(img_left)
            self.canvas.itemconfig(self.canvas_img, image=self.photo)

            self.photo2 = self.np_to_photo(img_right)
            self.canvas2.itemconfig(self.canvas_img2, image=self.photo2)

        self.root.after(30, self.update_gui)


# Start
threading.Thread(target=camera_loop, daemon=True).start()
root = tk.Tk()
app = App(root)
root.mainloop()
running = False
print("release resources...")
