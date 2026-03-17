import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge
from camera import Camera
from aruco import ARUCOBoardPose
from br import br_transform
import numpy as np


class ImageStream(Node):

    def __init__(self):
        super().__init__("image_stream")
        self.camleft = Camera(4, "./camera_param.yaml")
        self.board = ARUCOBoardPose()
        self.br = CvBridge()

        # message
        self.frame = "/camera_color_optical_frame"
        self.objframe = "/calib_board"
        self.tf_pub = TransformBroadcaster(self)
        self.infomsg = self.compose_camera_info()
        self.imgpub = self.create_publisher(Image, "/camera/color/image_raw", 1)
        self.infopub = self.create_publisher(
            CameraInfo, "/camera/color/camera_info", 1
        )
        self.timer = self.create_timer(0.01, callback=self.timer_callback)

    def timer_callback(self):
        timenow = self.get_clock().now().to_msg()
        ret, img = self.camleft.read()
        res = self.board.run(self.camleft, img)
        if res is not None:
            tvc, R = res
            H = np.eye(4)
            H[:3, :3] = R
            H[:3, 3] = tvc.flatten()
            self.get_logger().info(f"==>> cTo: \n{H}")
            q_wxyz = br_transform.to_quaternion(H)

            tfmsg = TransformStamped()
            tfmsg.header.stamp = timenow
            tfmsg.header.frame_id = self.frame.lstrip("/")
            tfmsg.child_frame_id = self.objframe.lstrip("/")
            tfmsg.transform.translation.x = float(H[0, 3])
            tfmsg.transform.translation.y = float(H[1, 3])
            tfmsg.transform.translation.z = float(H[2, 3])
            tfmsg.transform.rotation.x = float(q_wxyz[1])
            tfmsg.transform.rotation.y = float(q_wxyz[2])
            tfmsg.transform.rotation.z = float(q_wxyz[3])
            tfmsg.transform.rotation.w = float(q_wxyz[0])
            self.tf_pub.sendTransform(tfmsg)

        imgmsg = self.br.cv2_to_imgmsg(img, encoding="bgr8")
        imgmsg.header.stamp = timenow
        imgmsg.header.frame_id = self.frame
        self.imgpub.publish(imgmsg)
        self.infomsg.header.stamp = timenow
        self.infopub.publish(self.infomsg)

    def compose_camera_info(self):
        infomsg = CameraInfo()
        infomsg.header.frame_id = self.frame
        infomsg.height = self.camleft.info["height"]
        infomsg.width = self.camleft.info["width"]
        infomsg.distortion_model = self.camleft.info["distm"]
        infomsg.d = self.camleft.info["d"].tolist()
        infomsg.k = self.camleft.info["k"].flatten().tolist()
        infomsg.r = self.camleft.info["r"].flatten().tolist()
        infomsg.p = self.camleft.info["p"].flatten().tolist()
        return infomsg


def main(args=None):
    rclpy.init(args=args)
    imageNode = ImageStream()
    try:
        rclpy.spin(imageNode)
    except KeyboardInterrupt:
        imageNode.camleft.release()
    finally:
        imageNode.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
