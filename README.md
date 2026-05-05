# Handeye Calibration Lite
Lightweight packages for robotic manipulator and camera calibration

## Support Devices
- V4L2 Camera: Realsense, Webcam `(run on cv2_capture)`
- Zed Camera `(run on zed_sdk)`
- UR5e `(run on ur_rtde)`

## Support Calibration Mode
- Eye-in-hand: Camera on robot end-effector / `TF from Camera Frame to EE Frame`
- Eye-to-hand: Camera fixed on the world / `TF from Camera Frame to robot Base Frame`
- Mono Camera Process: Calibrate for single camera `K (intrinsic)` and `D (distortion)`
- Stereo Camera Process: Calibrate `TF from Camera Right To Camera Left` and `P (projection matrix)`


#### References
- https://docs.opencv.org/4.9.0/db/da9/tutorial_aruco_board_detection.html
