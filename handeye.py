import cv2
import yaml
import numpy as np
import threading
import time
import tkinter as tk
import rtde_control
import rtde_receive
from aruco import ARUCOBoardPose
from camera import Camera
import calibrator
import br
import solver

lock = threading.Lock()
running = True

# global variables to store latest poses
frame = None
latest_cTo_H = None # convention: cTo mean "obj to cam". read in reverse order
latest_bTe_H = None # convention: bTe mean "ee to base". read in reverse order
samples = []
result_matrix = None
result_quaternion = None

camera = Camera(4, "./camera_param.yaml")
board = ARUCOBoardPose()
hostip = "192.168.0.39"
rtde_c = rtde_control.RTDEControlInterface(hostip)
rtde_r = rtde_receive.RTDEReceiveInterface(hostip)


# Camera thread
def camera_loop():
    global frame, running, latest_cTo_H

    while running:

        ret, img = camera.read()

        res = board.run(camera, img)
        if res is not None:
            tvc, R = res
            H = np.eye(4)
            H[:3, :3] = R
            H[:3, 3] = tvc.flatten()
            with lock:
                latest_cTo_H = H

        if not ret:
            continue

        with lock:
            frame = img.copy()

    camera.release()


# Robot thread
def robot_loop():
    global running, latest_bTe_H

    rtde_c.teachMode()

    try:
        while running:
            tcp = rtde_r.getActualTCPPose()
            Rtcp, _ = cv2.Rodrigues(np.array(tcp[3:6]))
            H = np.eye(4)
            H[:3, :3] = Rtcp
            H[:3, 3] = tcp[0:3]
            with lock:
                latest_bTe_H = H
            time.sleep(0.01)  # Prevent CPU spinning
    finally:
        rtde_c.endTeachMode()


# GUI
class App:

    def __init__(self, root):
        self.root = root
        self.root.title("Hand-eye tool")
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        # buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack()

        tk.Button(btn_frame, text="Capture Sample", command=self.capture).pack(
            side="left"
        )
        tk.Button(btn_frame, text="Calibrate", command=self.calibrate).pack(
            side="left"
        )
        tk.Button(btn_frame, text="Reset", command=self.reset).pack(side="left")
        tk.Button(btn_frame, text="Save", command=self.save).pack(side="left")
        tk.Button(btn_frame, text="Quit", command=self.quit).pack(side="left")

        # text output area
        self.text = tk.Text(root, height=10, width=80)
        self.text.pack()

        # calibration mode
        mode_frame = tk.Frame(root)
        mode_frame.pack()

        tk.Label(mode_frame, text="Mode:").pack(side="left")

        self.mode = tk.StringVar(value="eye_in_hand")

        tk.Radiobutton(
            mode_frame, text="Eye-in-Hand", variable=self.mode, value="eye_in_hand"
        ).pack(side="left")

        tk.Radiobutton(
            mode_frame, text="Eye-to-Hand", variable=self.mode, value="eye_to_hand"
        ).pack(side="left")

        # save path
        path_frame = tk.Frame(root)
        path_frame.pack()

        tk.Label(path_frame, text="Save path:").pack(side="left")

        self.save_path = tk.Entry(path_frame, width=50)
        self.save_path.insert(0, "./handeye_result.yaml")
        self.save_path.pack(side="left")

        # sample counter
        self.status = tk.Label(root, text="Samples: 0")
        self.status.pack()

        self.update_gui()

    def log(self, msg):
        self.text.insert(tk.END, msg + "\n")
        self.text.see(tk.END)

    def capture(self):
        global latest_cTo_H, latest_bTe_H, samples

        with lock:
            if latest_cTo_H is None or latest_bTe_H is None:
                self.log("pose not ready")
                return
            # Create copies to avoid reference issues
            cTo = latest_cTo_H.copy()
            bTe = latest_bTe_H.copy()
            samples.append((bTe, cTo))

        self.log(f"capture sample #{len(samples)}")
        self.status.config(text=f"Samples: {len(samples)}")

    def calibrate(self):
        global samples, result_matrix, result_quaternion

        self.log("run calibration...!!!")

        with lock:
            samples_copy = [s for s in samples]  # Copy samples list
            if not samples_copy:
                self.log("no samples to calibrate")
                return

        if self.mode.get() == "eye_in_hand":
            self.log("mode: Eye-in-Hand")
            solver_cri = calibrator.HandEyeCalibrator(setup="Moving")
            result_name = "eTc"
        elif self.mode.get() == "eye_to_hand":
            self.log("mode: Eye-to-Hand")
            solver_cri = calibrator.HandEyeCalibrator(setup="Fixed")
            result_name = "bTc"
        else:
            self.log(f"unknown mode: {self.mode.get()}")
            return
        for sample in samples_copy:
            solver_cri.add_sample(sample[0], sample[1])

        X = solver_cri.solve(method=solver.Daniilidis1999)

        # baldor module give quternion in format qw qx qy qz
        X_q = br.br_transform.to_quaternion(X)
        XPose = [
            X[0, 3],
            X[1, 3],
            X[2, 3],
            X_q[1],
            X_q[2],
            X_q[3],
            X_q[0],
        ]

        with lock:
            result_matrix = X
            result_quaternion = XPose

        self.log(f"calibrated {result_name} in Matrix:\n{X}")
        self.log(
            f"calibrated {result_name} in quaternion [x y z qx qy qz qw]:\n{XPose}"
        )

    def reset(self):
        global samples
        with lock:
            samples.clear()
        self.status.config(text="Samples: 0")
        self.log("samples reset")

    def save(self):
        global result_matrix, result_quaternion
        path = self.save_path.get()

        with lock:
            if result_matrix is None or result_quaternion is None:
                self.log("no result yet!!!")
                return
            res_matrix = result_matrix.copy()
            res_quaternion = result_quaternion[:]

        res_matrix_list = res_matrix.astype(float).flatten().tolist()
        res_quaternion_list = [float(x) for x in res_quaternion]
        data = {
            "Calibration Mode": self.mode.get(),
            "Result in Matrix form (row major)": res_matrix_list,
            "Result in Quaternion form (xyz|qxqyqzqw)": res_quaternion_list,
        }
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        self.log(f"saved to {path}")

    def quit(self):
        global running
        running = False
        self.root.destroy()

    def update_gui(self):

        global frame

        img = None

        with lock:
            if frame is not None:
                img = frame.copy()

        if img is not None:

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape

            ppm = f"P6 {w} {h} 255 ".encode() + img.tobytes()
            photo = tk.PhotoImage(data=ppm)

            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.canvas.image = photo

        self.root.after(33, self.update_gui)


# Start threads
threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=robot_loop, daemon=True).start()
root = tk.Tk()
app = App(root)
root.mainloop()
running = False
print("release resources...")
