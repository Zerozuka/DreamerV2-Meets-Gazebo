#!/usr/bin/env python3
"""/result_img に流れる圧縮画像を OpenCV のウィンドウに表示する。

ROS 1 (rospy) から ROS 2 (rclpy) へ移植したもの。対応は次のとおり。

    rospy.init_node(name, anonymous=True)  -> rclpy.init() + Node(name)
    rospy.Subscriber(topic, T, cb)         -> node.create_subscription(T, topic, cb, qos)
    rospy.spin()                           -> rclpy.spin(node)

cv_bridge は使っていない。msg.data をそのまま cv2.imdecode に渡す
元の実装を踏襲している (CompressedImage なので bridge を通す必要がない)。
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage

WINDOW_NAME = "Result Image"


class ResultImageViewer(Node):

    def __init__(self):
        super().__init__("result_img_viewer")

        # 画像は取りこぼしても構わないので best effort。depth は 1 で
        # 最新フレームだけを見る (表示が遅れて溜まるのを避ける)。
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        self.create_subscription(
            CompressedImage, "/result_img", self._callback, qos
        )
        self.get_logger().info("subscribing to /result_img")

    def _callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            # 壊れたフレームで落ちないようにする。元の実装はここで
            # cv2.imshow に None を渡して例外になっていた。
            self.get_logger().warn("failed to decode a frame, skipping")
            return
        cv2.imshow(WINDOW_NAME, image)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ResultImageViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        # spin が例外で抜けた場合 rclpy がまだ生きていることがある。
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
