#!/usr/bin/env python3
"""ROS 1 の rospy 風 API を rclpy の上に再現する移植用の薄い層。

なぜこれを置くか
----------------
JEPA の学習・評価スクリプト 15 本は ROS 1 の「グローバルノード」慣用句で
書かれている。モジュールレベルに rospy.Subscriber を並べ、コールバックは
自由関数として定義する形である。

    rospy.init_node('Gazebo_test', anonymous=True)
    rospy.Subscriber("/channels", Float32MultiArray, channel_callback)
    flag_pub = rospy.Publisher('/render_trigger', Int32, queue_size=10)

rclpy にグローバルノードは存在しないため、本来は各スクリプトで Node を作り、
コールバックをメソッドにするのが筋である。ただしこれらのスクリプトは学習済み
重みが無いと実行できず、移植の正しさを実行して確かめられない。15 本を未検証で
書き換えるより、意味を厳密に保つ層を 1 つ用意して呼び出し側を無改修に保つ方が
リスクが低いと判断した。

したがってこれは恒久的な設計ではなく移植の足場である。スクリプトごとに
Node を持つ形へ整理する際は、この層を外していく。

rospy との違い
--------------
- init_node() が executor を daemon スレッドで起動する。rclpy は executor が
  spin しない限りコールバックが呼ばれないため。
- init_node() を呼ぶ前に Subscriber / Publisher を作っても暗黙に初期化する。
  ROS 1 でも順序を守るのが作法だったが、守っていないスクリプトがあるため。
- sleep() は use_sim_time が有効なら sim 時間で待つ。sim が止まったときに
  固まらないよう壁時計の上限を設けている。

使い方 (呼び出し側の差分を最小にするため別名で import する)

    import ros1_compat as rospy
"""

import threading
import time

import rclpy
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

#: init_node() が作るプロセス共有のノード。
_node = None
_executor = None
_spin_thread = None
_lock = threading.Lock()


class ROSInterruptException(Exception):
    """rospy.ROSInterruptException の代替。

    rclpy は spin 中断時に KeyboardInterrupt を投げるので、これが送出される
    ことはない。except 節を書いているスクリプトが動くように名前だけ用意する。
    """


def _ensure_node(name="ros1_compat"):
    """ノードと executor を用意する。既にあれば何もしない。"""
    global _node, _executor, _spin_thread
    with _lock:
        if _node is not None:
            return _node
        if not rclpy.ok():
            rclpy.init()
        _node = Node(name)
        # rospy.sleep は /use_sim_time が有効なとき sim 時間で待っていた。
        # sleep() が同じ意味になるようクロックを合わせる。
        _node.set_parameters([
            rclpy.parameter.Parameter(
                "use_sim_time", rclpy.Parameter.Type.BOOL, True)
        ])
        _executor = SingleThreadedExecutor()
        _executor.add_node(_node)
        _spin_thread = threading.Thread(target=_executor.spin, daemon=True)
        _spin_thread.start()
        return _node


def init_node(name, anonymous=False, **_ignored):
    """rospy.init_node の代替。

    anonymous は rclpy では意味を持たないが呼び出し側が渡しているので受け取って
    無視する。ノード名が重複して困る場合は呼び出し側で名前を変えること。
    """
    return _ensure_node(name)


def get_node():
    """内部のノードを取り出す。rclpy の API を直接使いたいとき用。"""
    return _ensure_node()


def Subscriber(topic, msg_type, callback, queue_size=10, **_ignored):
    """rospy.Subscriber の代替。

    rospy は戻り値を保持しなくても購読が維持された。rclpy も
    create_subscription の結果はノードが保持するので同じ挙動になる。
    """
    node = _ensure_node()
    return node.create_subscription(msg_type, topic, callback, queue_size)


def Publisher(topic, msg_type, queue_size=10, **_ignored):
    """rospy.Publisher の代替。publish() を持つオブジェクトを返す。"""
    node = _ensure_node()
    return node.create_publisher(msg_type, topic, queue_size)


def is_shutdown():
    """rospy.is_shutdown の代替。"""
    return not rclpy.ok()


def spin():
    """rospy.spin の代替。

    executor は既に別スレッドで回っているので、ここでは中断まで待つだけ。
    """
    _ensure_node()
    try:
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


def sleep(seconds):
    """rospy.sleep の代替。use_sim_time が有効なら sim 時間で待つ。

    sim が一時停止すると sim 時間が進まず永久に待つことになるので、壁時計の
    上限を設けて抜ける。上限は要求の 20 倍か 5 秒の大きい方。
    """
    node = _ensure_node()
    clock = node.get_clock()
    end = clock.now() + Duration(seconds=seconds)
    wall_limit = max(seconds * 20.0, 5.0)
    wall_start = time.monotonic()
    while clock.now() < end:
        if time.monotonic() - wall_start > wall_limit:
            node.get_logger().warn(
                f"sleep({seconds}) が sim 時間で完了しなかった "
                f"(壁時計 {wall_limit:.1f} 秒で打ち切り)。"
                f"Gazebo が止まっているか /clock が来ていない。")
            return
        time.sleep(0.001)


def _fmt(msg, args):
    return str(msg) % args if args else str(msg)


def loginfo(msg, *args):
    _ensure_node().get_logger().info(_fmt(msg, args))


def logwarn(msg, *args):
    _ensure_node().get_logger().warn(_fmt(msg, args))


def logerr(msg, *args):
    _ensure_node().get_logger().error(_fmt(msg, args))


def logdebug(msg, *args):
    _ensure_node().get_logger().debug(_fmt(msg, args))


def shutdown(reason=""):
    """rospy.shutdown の代替。executor を止めてノードを破棄する。

    順序が重要である。executor を止める前にノードを破棄すると、spin 中の
    executor が破棄済みのオブジェクトに触って次の通知が出る。

      The following exception was never retrieved:
      cannot use Destroyable because destruction was requested

    害はないが紛らわしいので、executor の停止とスレッドの合流を待ってから
    ノードを破棄する。
    """
    global _node, _executor, _spin_thread
    with _lock:
        if _node is None:
            return
        node, executor, thread = _node, _executor, _spin_thread
        # 先に参照を外し、他スレッドから _ensure_node が触らないようにする
        _node = _executor = _spin_thread = None

    try:
        executor.shutdown(timeout_sec=2.0)
    except Exception:
        pass
    if thread is not None and thread.is_alive():
        thread.join(timeout=2.0)
    try:
        executor.remove_node(node)
    except Exception:
        pass
    try:
        node.destroy_node()
    except Exception:
        pass
    if rclpy.ok():
        rclpy.shutdown()


class Time:
    """rospy.Time.now() を使っているコード向けの最小の代替。"""

    @staticmethod
    def now():
        return _ensure_node().get_clock().now()
