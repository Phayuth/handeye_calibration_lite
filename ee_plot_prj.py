import cv2
import numpy as np
import time

# canvas
H, W = 500, 500
img = np.zeros((H, W, 3), dtype=np.uint8)

# fake TCP history
tcp_history = []
MAX_POINTS = 300

# projection parameters
scale = 200.0   # meters -> pixels
cx = W // 2
cy = H // 2


def project_xy(p):
    x = int(cx + p[0] * scale)
    y = int(cy - p[1] * scale)
    return x, y


def draw_path():

    img[:] = 0

    if len(tcp_history) < 2:
        return

    for i in range(1, len(tcp_history)):
        p1 = project_xy(tcp_history[i - 1])
        p2 = project_xy(tcp_history[i])
        cv2.line(img, p1, p2, (0, 255, 0), 2)

    cv2.circle(img, project_xy(tcp_history[-1]), 5, (0, 0, 255), -1)

    cv2.putText(img, "Top view (X,Y)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)


# initial fake pose
p = np.array([0.0, 0.0, 0.0])

while True:

    # fake smooth robot motion
    step = np.random.randn(3) * 0.002
    p = p + step

    tcp_history.append(p.copy())

    if len(tcp_history) > MAX_POINTS:
        tcp_history.pop(0)

    draw_path()

    cv2.imshow("Robot EE Path", img)

    if cv2.waitKey(30) == 27:
        break

    time.sleep(0.03)

cv2.destroyAllWindows()