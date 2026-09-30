#!/usr/bin/env python3
"""world に固定設置したカメラの映像を並べて表示する。

ROS 1 (rospy) から ROS 2 (rclpy) へ移植したもの。

    rospy.init_node(name)          -> rclpy.init() + Node(name)
    rospy.Subscriber(t, T, cb)     -> node.create_subscription(T, t, cb, qos)
    rospy.spin()                   -> rclpy.spin(node)

トピック名について
------------------
ROS 1 版は 5 台のカメラが別々のトピックに出ている前提で
/cam_front/world_cam/image_raw のように購読していた。

Gazebo Harmonic では world_camera/model.sdf の <sensor><topic> が
include 時の <name> で上書きされないため、cam_front / cam_back / cam_left /
cam_right / cam_top の 5 台すべてが同じ world_cam/image_raw に publish する。
そのため既定では 1 トピックのみを購読する。

台ごとに分けたい場合は world_camera を複製して <topic> を変え、下の
CAMERA_TOPICS を書き換える。あわせて ros_gz_bridge で ROS 側に出す必要がある
(jetbot_tellus.launch.py の bridge には未登録)。
"""

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image

#: 表示するウィンドウ名とトピックの対応。
CAMERA_TOPICS = {
    "world_cam": "/world_cam/image_raw",
}


class MultiCameraViewer(Node):

    def __init__(self):
        super().__init__("multi_camera_viewer")
        self.bridge = CvBridge()

        # 画像は取りこぼしても構わないので best effort、最新フレームのみ。
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        for name, topic in CAMERA_TOPICS.items():
            self.create_subscription(
                Image, topic, self._make_callback(name), qos)
            self.get_logger().info(f"subscribing to {topic} as '{name}'")

    def _make_callback(self, name):
        def callback(msg):
            try:
                img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception as e:
                # 壊れたフレームで落ちないようにする。
                self.get_logger().warn(f"{name}: failed to convert frame: {e}")
                return
            cv2.imshow(name, img)
            cv2.waitKey(1)
        return callback


def main(args=None):
    rclpy.init(args=args)
    node = MultiCameraViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
