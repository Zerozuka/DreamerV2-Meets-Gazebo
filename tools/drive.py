#!/usr/bin/env python3
"""Drive the robot from the keyboard, and point the Gazebo GUI camera at it.

The Jetbot is about 17 cm long and spawns in a 40 m arena, so the default
Gazebo camera shows the whole hall with the robot as a few grey pixels. This
script covers both halves of "watch the robot move": it asks the GUI to follow
the robot, and it publishes /cmd_vel from the keyboard.

    source /opt/ros/jazzy/setup.bash
    source ~/ros2_ws/install/setup.bash
    cd ~/ros2_ws/src/DreamerV2-Meets-Gazebo
    .venv/bin/python tools/drive.py

Keys:

    up / w      forward            down / s    backward
    left / a    turn left          right / d   turn right
    space       stop               q           quit
    + / -       faster / slower

Pass --no-follow to leave the GUI camera alone, which is what you want when
running headless or when you have already framed the shot by hand.
"""

import argparse
import math
import os
import select
import subprocess
import sys
import termios
import time
import tty

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

# 押しっぱなしを模すため、キーが来ない間も最後の指令を送り続ける。離してから
# この秒数で停止する。短すぎるとカクつき、長すぎると止まらなくなる。
HOLD = 0.4

KEYS = {
    "w": (1, 0), "s": (-1, 0), "a": (0, 1), "d": (0, -1),
    "\x1b[A": (1, 0), "\x1b[B": (-1, 0), "\x1b[D": (0, 1), "\x1b[C": (0, -1),
}


def gz_service(service, reqtype, req):
    """gz service を呼ぶ。GUI が動いていない場合もあるので失敗は無視する。"""
    try:
        subprocess.run(
            ["gz", "service", "-s", service, "--reqtype", reqtype,
             "--reptype", "gz.msgs.Boolean", "--timeout", "3000", "--req", req],
            capture_output=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def start_following(model, back, up):
    """GUI のカメラを model に追従させる。GUI が無ければ何も起きない。"""
    gz_service("/gui/follow", "gz.msgs.StringMsg", f'data: "{model}"')
    gz_service("/gui/follow/offset", "gz.msgs.Vector3d",
               f"x: {-abs(back)}, y: 0.0, z: {abs(up)}")


def shutdown(node):
    """停止指令を送ってから後片付けする。

    Ctrl-C や timeout で rclpy が先に落ちていることがあるので、二重 shutdown で
    RCLError を出さないように状態を見てから呼ぶ。
    """
    node.send(0, 0)
    time.sleep(0.1)
    try:
        node.destroy_node()
    except Exception:  # noqa: BLE001 - 後片付けなので失敗しても続ける
        pass
    if rclpy.ok():
        rclpy.shutdown()
    print("\nstopped")


class Driver(Node):
    def __init__(self, linear, angular):
        super().__init__("drive")
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(Odometry, "/odom", self._odom, 10)
        self.linear = linear
        self.angular = angular
        self.pose = None

    def _odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y ** 2 + q.z ** 2))
        self.pose = (p.x, p.y, math.degrees(yaw))

    def send(self, fwd, turn):
        # Ctrl-C で rclpy が落ちた後に finally から呼ばれることがある。
        # その状態で publish すると context invalid で例外になるので弾く。
        if not rclpy.ok():
            return
        msg = Twist()
        msg.linear.x = fwd * self.linear
        msg.angular.z = turn * self.angular
        self.pub.publish(msg)


def read_key(timeout):
    """キーを 1 つ読む。矢印キーは 3 バイトのエスケープ列なのでまとめて読む。"""
    if not select.select([sys.stdin], [], [], timeout)[0]:
        return None
    ch = sys.stdin.read(1)
    if ch == "\x1b" and select.select([sys.stdin], [], [], 0.01)[0]:
        ch += sys.stdin.read(2)
    return ch


def load_course():
    """gz_sionna/config/path_points.csv を読む。(x, y) の列を返す。

    列は x, y, yaw の 3 つで yaw はラジアン。ここでは位置だけ使い、向きは
    追従制御が決める。
    """
    import csv
    from ament_index_python.packages import get_package_share_directory

    path = os.path.join(get_package_share_directory("gz_sionna"),
                        "config", "path_points.csv")
    with open(path, encoding="utf-8") as fh:
        return [(float(r["x"]), float(r["y"])) for r in csv.DictReader(fh)]


def run_path(node, lookahead, laps):
    """コースに沿って走る。pure pursuit で前方の点を追いかけるだけ。

    jetbot_tellus.launch.py の既定スポーン位置はこのコース上にあるので、
    そのまま起動して呼べば壁に当たらずに走れる。
    """
    course = load_course()
    print(f"following the course: {len(course)} points, "
          f"lookahead {lookahead:.2f} m, "
          f"{laps if laps else 'unlimited'} lap(s). Ctrl-C to stop.")

    # 最初の odom を待ってから、いま一番近い点を目標にする。0 番から始めると
    # スポーン位置 (コース上の 11 番付近) の後ろを向こうとしてその場で回り、
    # 壁に当たる。
    while rclpy.ok() and not node.pose:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not node.pose:
        print("no /odom: is Gazebo running?", file=sys.stderr)
        return
    x0, y0, _ = node.pose
    target = min(range(len(course)),
                 key=lambda i: math.hypot(course[i][0] - x0, course[i][1] - y0))
    print(f"starting from point #{target} "
          f"({course[target][0]:.2f}, {course[target][1]:.2f})")

    lap = 0
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if not node.pose:
                continue
            x, y, yaw_deg = node.pose
            yaw = math.radians(yaw_deg)

            # lookahead より遠い点が見つかるまで目標を進める。一周したら戻る。
            steps = 0
            while steps < len(course):
                tx, ty = course[target]
                if math.hypot(tx - x, ty - y) > lookahead:
                    break
                target = (target + 1) % len(course)
                steps += 1
                if target == 0:
                    lap += 1
                    if laps and lap >= laps:
                        return

            tx, ty = course[target]
            # ロボット座標系での目標方向。alpha が 0 なら真っ直ぐ前。
            alpha = math.atan2(ty - y, tx - x) - yaw
            alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
            dist = math.hypot(tx - x, ty - y)

            # 目標が真横より後ろにあるときは前進せず、その場で向きを合わせる。
            if abs(alpha) > math.pi / 2:
                fwd, turn = 0.0, math.copysign(1.0, alpha)
            else:
                fwd = 1.0
                # pure pursuit: omega = 2 v sin(alpha) / L。node.send が
                # angular に node.angular を掛けるので -1..1 に正規化して渡す。
                turn = max(-1.0, min(1.0,
                                     2 * math.sin(alpha) / max(lookahead, 0.1)))
            node.send(fwd, turn)

            sys.stdout.write(
                f"\rx={x:7.2f}  y={y:7.2f}  yaw={yaw_deg:7.1f}deg   "
                f"target #{target:3d} ({tx:6.2f},{ty:6.2f}) {dist:5.2f} m   "
                f"lap {lap}   ")
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        shutdown(node)


def run_demo(node, seconds):
    """キー入力なしで走り回る。ただ動いているところを見たいときに使う。

    前進と旋回を交互に繰り返す。seconds が 0 なら Ctrl-C まで続ける。
    """
    # (前進量, 旋回量, 秒数) の繰り返し。壁にぶつかっても向きを変えて抜ける。
    pattern = [(1, 0, 4.0), (0, 1, 2.0), (1, 0, 4.0), (0, -1, 2.0)]
    print("demo: forward and turn, repeating. Ctrl-C to stop.")
    end = time.time() + seconds if seconds else None
    try:
        while rclpy.ok():
            for fwd, turn, duration in pattern:
                stop = time.time() + duration
                while time.time() < stop and rclpy.ok():
                    if end and time.time() > end:
                        return
                    node.send(fwd, turn)
                    rclpy.spin_once(node, timeout_sec=0.05)
                    if node.pose:
                        x, y, yaw = node.pose
                        sys.stdout.write(
                            f"\rx={x:7.2f}  y={y:7.2f}  yaw={yaw:7.1f}deg   "
                            f"v={fwd * node.linear:+.2f} m/s  "
                            f"w={turn * node.angular:+.2f} rad/s   ")
                        sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        shutdown(node)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="jetbot_1",
                        help="model name to follow (default: jetbot_1)")
    parser.add_argument("--no-follow", action="store_true",
                        help="do not touch the GUI camera")
    parser.add_argument("--back", type=float, default=1.2,
                        help="metres behind the robot for the camera")
    parser.add_argument("--up", type=float, default=0.6,
                        help="metres above the robot for the camera")
    parser.add_argument("--speed", type=float, default=0.4,
                        help="forward speed in m/s (default: 0.4)")
    parser.add_argument("--turn", type=float, default=1.0,
                        help="turn rate in rad/s (default: 1.0)")
    parser.add_argument("--demo", metavar="SECONDS", nargs="?", type=float,
                        const=0.0, default=None,
                        help="drive a fixed pattern instead of reading keys; "
                             "runs for SECONDS, or until Ctrl-C if omitted")
    parser.add_argument("--path", metavar="LAPS", nargs="?", type=int,
                        const=0, default=None,
                        help="follow the course in gz_sionna/config/"
                             "path_points.csv for LAPS laps, or until Ctrl-C")
    parser.add_argument("--lookahead", type=float, default=0.6,
                        help="pure pursuit lookahead in metres (default: 0.6)")
    args = parser.parse_args()

    if not args.no_follow:
        print(f"pointing the Gazebo GUI camera at {args.model} ...")
        start_following(args.model, args.back, args.up)

    rclpy.init()
    node = Driver(args.speed, args.turn)

    if args.path is not None:
        run_path(node, args.lookahead, args.path)
        return

    if args.demo is not None:
        run_demo(node, args.demo)
        return

    if not sys.stdin.isatty():
        print("This needs a real terminal to read the keyboard.\n"
              "Run it directly in a terminal, or use --demo to drive a fixed "
              "pattern without any input.", file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(2)

    print("""
    up / w      forward            down / s    backward
    left / a    turn left          right / d   turn right
    space       stop               q           quit
    + / -       faster / slower
""")

    settings = termios.tcgetattr(sys.stdin)
    fwd = turn = 0
    last = 0.0
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            key = read_key(0.05)
            if key:
                if key in ("q", "\x03"):
                    break
                if key == " ":
                    fwd = turn = 0
                elif key in KEYS:
                    fwd, turn = KEYS[key]
                    last = time.time()
                elif key in "+=":
                    node.linear = min(node.linear + 0.1, 2.0)
                    node.angular = min(node.angular + 0.2, 4.0)
                elif key == "-":
                    node.linear = max(node.linear - 0.1, 0.1)
                    node.angular = max(node.angular - 0.2, 0.2)

            if (fwd or turn) and time.time() - last > HOLD:
                fwd = turn = 0

            node.send(fwd, turn)
            rclpy.spin_once(node, timeout_sec=0.0)

            if node.pose:
                x, y, yaw = node.pose
                sys.stdout.write(
                    f"\rx={x:7.2f}  y={y:7.2f}  yaw={yaw:7.1f}deg   "
                    f"v={fwd * node.linear:+.2f} m/s  "
                    f"w={turn * node.angular:+.2f} rad/s   ")
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        shutdown(node)


if __name__ == "__main__":
    main()
