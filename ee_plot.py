import cv2
import numpy as np
import rtde_control
import rtde_receive

WINDOW_NAME = "3D Frames View - Camera Perspective"
IMAGE_HEIGHT = 250
IMAGE_WIDTH = 250
AXIS_LENGTH = 0.5
FRAME_DELAY_MS = 30

# Canvas setup
img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

hostip = "192.168.0.39"
rtde_c = rtde_control.RTDEControlInterface(hostip)
rtde_r = rtde_receive.RTDEReceiveInterface(hostip)

# Camera intrinsic parameters (simulated camera)
focal_length = 300  # pixels
cx = IMAGE_WIDTH // 2
cy = IMAGE_HEIGHT // 2
K = np.array(
    [
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1],
    ]
)
dist_coeffs = np.zeros(5)  # no lens distortion


def look_at_rotation(camera_pos, target_pos, up=np.array([0.0, 0.0, 1.0])):
    """make rotation so camera looks at target."""
    z_cam_world = target_pos - camera_pos
    z_cam_world /= np.linalg.norm(z_cam_world)
    x_cam_world = np.cross(z_cam_world, up)
    if np.linalg.norm(x_cam_world) < 1e-8:
        up = np.array([0.0, 1.0, 0.0])
        x_cam_world = np.cross(z_cam_world, up)
    x_cam_world /= np.linalg.norm(x_cam_world)
    y_cam_world = np.cross(z_cam_world, x_cam_world)
    y_cam_world /= np.linalg.norm(y_cam_world)
    R_wc = np.column_stack([x_cam_world, y_cam_world, z_cam_world])
    return R_wc.T


# Frame A is the world/base frame
# Camera pose relative to frame A/world: C = (1, 1, 1), looking at A origin
t_A_world = np.array([0.0, 0.0, 0.0])
camera_pos_world = np.array([1.0, 1.0, 1.0])
R_cw = look_at_rotation(camera_pos_world, t_A_world)
t_cw = -R_cw @ camera_pos_world

# Frame A is fixed in the world, so its camera pose never changes.
R_A_cam = R_cw
t_A_cam = t_cw
rvec_A, _ = cv2.Rodrigues(R_A_cam)


def get_tcp_pose_world():
    """Return the current TCP rotation and translation in the base frame."""
    tcp = rtde_r.getActualTCPPose()
    rvec_tcp = np.asarray(tcp[3:6], dtype=float)
    R_tcp_world, _ = cv2.Rodrigues(rvec_tcp)
    t_tcp_world = np.asarray(tcp[0:3], dtype=float)
    return R_tcp_world, t_tcp_world


def draw_frames(R_B_world, t_B_world):
    """Draw the fixed world (frame A) and the moving TCP (frame B) frame."""
    img[:] = 255

    # Frame B in camera coordinates
    R_B_cam = R_cw @ R_B_world
    t_B_cam = R_cw @ t_B_world + t_cw
    rvec_B, _ = cv2.Rodrigues(R_B_cam)

    # world (frame A)
    cv2.drawFrameAxes(img, K, dist_coeffs, rvec_A, t_A_cam, AXIS_LENGTH)

    # TCP (frame B)
    cv2.drawFrameAxes(
        img, K, dist_coeffs, rvec_B, t_B_cam, AXIS_LENGTH - 0.2, thickness=2
    )


def main():
    rtde_c.teachMode()

    try:
        while True:
            R_tcp_world, t_tcp_world = get_tcp_pose_world()
            draw_frames(R_tcp_world, t_tcp_world)
            cv2.imshow(WINDOW_NAME, img)

            if cv2.waitKey(FRAME_DELAY_MS) == 27:  # ESC to exit
                break
    except Exception as error:
        print(f"Error getting robot pose: {error}")
    finally:
        rtde_c.endTeachMode()
        cv2.destroyAllWindows()
        print("Program ended.")


if __name__ == "__main__":
    main()
