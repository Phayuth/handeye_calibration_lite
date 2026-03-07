import cv2
import numpy as np
import time

# Canvas setup
H, W = 500, 500
img = np.zeros((H, W, 3), dtype=np.uint8)

# Camera intrinsic parameters (simulated camera)
focal_length = 300  # pixels
cx = W // 2
cy = H // 2
K = np.array(
    [[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]], dtype=np.float64
)
dist_coeffs = np.zeros(5)  # no lens distortion

# Frame A is the world/base frame
R_A_world = np.eye(3, dtype=np.float64)
t_A_world = np.array([0.0, 0.0, 0.0], dtype=np.float64)


def look_at_rotation(camera_pos, target_pos, up=np.array([0.0, 0.0, 1.0])):
    """Return R_cw (world -> camera) so camera looks at target."""
    z_cam_world = target_pos - camera_pos
    z_cam_world /= np.linalg.norm(z_cam_world)

    x_cam_world = np.cross(z_cam_world, up)
    if np.linalg.norm(x_cam_world) < 1e-8:
        up = np.array([0.0, 1.0, 0.0])
        x_cam_world = np.cross(z_cam_world, up)
    x_cam_world /= np.linalg.norm(x_cam_world)

    y_cam_world = np.cross(z_cam_world, x_cam_world)
    y_cam_world /= np.linalg.norm(y_cam_world)

    # Camera axes expressed in world coordinates (columns)
    R_wc = np.column_stack([x_cam_world, y_cam_world, z_cam_world])
    return R_wc.T


# Camera pose relative to frame A/world: C = (1, 1, 1), looking at A origin
camera_pos_world = np.array([1.0, 1.0, 1.0], dtype=np.float64)
R_cw = look_at_rotation(camera_pos_world, t_A_world)
t_cw = -R_cw @ camera_pos_world

# Frame B pose in world (updated in loop)
t_B_world = np.array([0.5, 0.0, 0.0], dtype=np.float64)
R_B_world = np.eye(3, dtype=np.float64)

# Frame C is fixed to frame B with a constant transform B->C
t_BC = np.array([0.35, 0.2, 0.15], dtype=np.float64)
angle_BC = np.deg2rad(25.0)
R_BC = np.eye(3, dtype=np.float64)


def draw_frames():
    """Draw frame A, moving frame B, and frame C fixed to B."""
    img[:] = 0

    # Frame A in camera coordinates
    R_A_cam = R_cw @ R_A_world
    t_A_cam = R_cw @ t_A_world + t_cw

    # Frame B in camera coordinates
    R_B_cam = R_cw @ R_B_world
    t_B_cam = R_cw @ t_B_world + t_cw

    # Frame C in world and then camera coordinates (C fixed to B)
    R_C_world = R_B_world @ R_BC
    t_C_world = t_B_world + R_B_world @ t_BC
    R_C_cam = R_cw @ R_C_world
    t_C_cam = R_cw @ t_C_world + t_cw

    # Convert rotation matrices to rotation vectors for OpenCV
    rvec_A, _ = cv2.Rodrigues(R_A_cam)
    rvec_B, _ = cv2.Rodrigues(R_B_cam)
    rvec_C, _ = cv2.Rodrigues(R_C_cam)

    # Draw frame A (base) - axis length 0.5m
    cv2.drawFrameAxes(img, K, dist_coeffs, rvec_A, t_A_cam, 0.5)

    # Draw frame B (moving) - axis length 0.5m
    cv2.drawFrameAxes(img, K, dist_coeffs, rvec_B, t_B_cam, 0.5)

    # Draw frame C (fixed to B) - axis length 0.4m
    cv2.drawFrameAxes(img, K, dist_coeffs, rvec_C, t_C_cam, 0.4)

    # Add info text
    cv2.putText(
        img,
        "Frame A (Base) | Frame B (Moving) | Frame C (Fixed to B)",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
    )

    # Show distances from A
    distance = np.linalg.norm(t_B_world - t_A_world)
    distance_c = np.linalg.norm(t_C_world - t_A_world)
    cv2.putText(
        img,
        f"A->B: {distance:.2f}m | A->C: {distance_c:.2f}m",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )


# Animation parameters
step = 0

while True:
    step += 0.02

    # Move frame B in 3D space relative to frame A (max 2m away)
    # Circular motion with varying radius and height
    radius = 0.9 + 0.6 * np.sin(step * 0.3)  # radius varies 0.3 to 1.5m
    angle = step * 0.8

    # Calculate B's position relative to A
    offset_x = radius * np.cos(angle)
    offset_y = radius * np.sin(angle)
    offset_z = 0.3 * np.sin(step * 1.2)  # slight Z variation

    t_B_world = t_A_world + np.array([offset_x, offset_y, offset_z])

    # Rotate frame B around its own Z axis
    rot_angle = step * 0.5
    R_B_world = np.array(
        [
            [np.cos(rot_angle), -np.sin(rot_angle), 0],
            [np.sin(rot_angle), np.cos(rot_angle), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    draw_frames()
    cv2.imshow("3D Frames View - Camera Perspective", img)

    if cv2.waitKey(30) == 27:  # ESC to exit
        break

    time.sleep(0.03)

cv2.destroyAllWindows()
