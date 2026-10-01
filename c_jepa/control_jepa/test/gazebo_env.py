#!/usr/bin/env python3
import csv
import math
import os
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
import tf_transformations as tft
# cv_bridge は使わない。apt の cv_bridge_boost.so が NumPy 1 でコンパイル
# されており NumPy 2 (sionna 2.x が要求) では画像変換が KeyError で落ちる。
# 同等の変換を gz_sionna/src/ros_image.py に自前で持たせた (差分は docstring)。
from geometry_msgs.msg import Twist
from gymnasium import Env, spaces
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image

import matplotlib.pyplot as plt

# Gazebo Harmonic では gazebo_msgs の以下が使えない。gz-transport を叩く
# ラッパー (gz_sionna/src/gz_world_control.py) に置き換えた。
#
#   gazebo_msgs.srv.SetModelState        -> GzWorldControl.set_pose
#   gazebo_msgs.srv.GetPhysicsProperties -> 廃止 (max_step_size は設定値)
#   std_srvs.srv.Empty (pause/unpause)   -> GzWorldControl.pause / unpause
#
# gz_sionna はまだ Python モジュールを install していない (dreamerv2 / utils の
# 名前衝突を解消するまで packages=[] のため) ので、share のパスを ament_index
# 経由で解決して import する。衝突解消後は普通の import に直す。
from ament_index_python.packages import get_package_share_directory
sys.path.insert(0, os.path.join(get_package_share_directory('gz_sionna'), 'src'))
from ros_image import imgmsg_to_bgr8  # noqa: E402
from gz_world_control import GzWorldControl  # noqa: E402


def _resolve_data_file(env_var, filename, description):
    """コースデータ CSV のパスを解決する。

    元は開発者の home を指す絶対パス (/home/icon-group/catkin_ws/src/...) が
    ハードコードされていて、他のマシンでは動かなかった。次の順で探す。

      1. 環境変数 env_var        別のコースを使いたいとき
      2. gz_sionna の config/    同梱の Tellus コース (既定)
      3. カレントディレクトリ    手元で差し替えたいとき

    通常は 2 で解決するので、環境変数を設定しなくても GazeboEnv() が動く。
    """
    path = os.environ.get(env_var)
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{env_var}={path} が指すファイルが存在しない ({description})"
            )
        return path

    packaged = os.path.join(
        get_package_share_directory('gz_sionna'), 'config', filename)
    if os.path.exists(packaged):
        return packaged

    local = os.path.join(os.getcwd(), filename)
    if os.path.exists(local):
        return local

    raise FileNotFoundError(
        f"{description} が見つからない。\n"
        f"  想定した場所: {packaged}\n"
        f"  環境変数 {env_var} にパスを設定するか、\n"
        f"  カレントディレクトリに {filename} を置くこと。"
    )
import random
import torch

# MUD_COLOR = (14, 34, 49) 
MUD_COLOR = (0, 0, 0) 


class GazeboEnv(Env):
    def __init__(self, path_file=None, robot_ns=""):
        super(GazeboEnv, self).__init__()

        # Gazebo Harmonic の world 制御。Classic の /gazebo/* サービスの代替。
        self.gz = GzWorldControl(world=os.environ.get("GZ_WORLD", "default"))
        self.robot_model_name = os.environ.get("GZ_ROBOT_NAME", "jetbot_1")

        # === Load trajectory ===
        if path_file is None:
            path_file = _resolve_data_file(
                "JEPA_PATH_POINTS", "path_points.csv",
                "経路点の CSV (列 x, y, yaw)")
        self.path = self._load_path(path_file)
        self.lookahead_distance = 0.5
        self.goal_tolerance = 0.15
        # self.linear_speed =0.7 # 0.70
        self.target_index = 0
        self.robot_ns = robot_ns
        self.max_deviation = 0.50 # max lateral deviation from path
        self.render_mode = "human"

        self._current_speed = 0.0
        self._current_steer = 0.0
        self._last_time = None
        self._last_desired_steer = 0.0  # <-- new: steering memory


        self.last_move_time = time.time()
        self.last_position = None

        self.img_src = None
        self.brake_frames = 0
        self.brake_terminate_threshold = 15  # ~10–15 frames
        self.flag_no_obs = True
        self.visual_mode = 0



        # === ROS 2 ノードと executor ===
        #
        # rospy は Subscriber を作るだけでバックグラウンドスレッドが
        # コールバックを回してくれたが、rclpy は executor が spin しない限り
        # コールバックが一切呼ばれない。
        #
        # reset() は「コールバックで画像と odom が届くまで待つ」構造
        # (current_pose / current_image が None の間ループ) なので、
        # 単一スレッドで spin すると待ちループが executor をブロックして
        # 永久にデッドロックする。
        #
        # そこで executor を別スレッドで回し、step() / reset() は従来どおり
        # 同期的に書けるようにしている。
        #
        # SingleThreadedExecutor で足りる。デッドロックを避けるのに必要なのは
        # 「executor が待ちループとは別のスレッドで回ること」だけで、
        # コールバック同士の並行性は使っていない (どちらも参照の代入のみ)。
        # MultiThreadedExecutor は内部の ThreadPoolExecutor が終了時に
        # "cannot use Destroyable because destruction was requested" を
        # 出すので避けている。
        #
        # コールバックは executor スレッド側で走るため、_odom_callback /
        # _image_callback が書き込む current_pose / current_image には別
        # スレッドから触ることになる。参照の代入だけなので GIL の範囲で安全
        # (部分更新は起きない)。
        if not rclpy.ok():
            rclpy.init()

        node_name = f"gazebo_env{robot_ns.replace('/', '_')}" if robot_ns else "gazebo_env"
        self.node = Node(node_name)
        self._owns_context = True

        # rospy.sleep は /use_sim_time が有効なとき sim 時間で待っていた。
        # _sleep() が同じ意味になるよう、ノードのクロックを sim 時間に切り替える。
        # これを設定しないと get_clock() が壁時計を返し、実時間係数が 1 を
        # 下回る環境では 1 step で進む sim 時間が想定より短くなる。
        #
        # 環境変数 JEPA_USE_SIM_TIME=0 で無効化できる。Gazebo を動かさずに
        # 報酬ロジックだけ試すときなど、/clock が来ない状況では壁時計の方が
        # 都合がよい (sim 時間だと _sleep が打ち切りまで待つ)。
        use_sim_time = os.environ.get("JEPA_USE_SIM_TIME", "1") not in ("0", "false", "False")
        self.node.set_parameters([
            rclpy.parameter.Parameter(
                "use_sim_time", rclpy.Parameter.Type.BOOL, use_sim_time)
        ])

        # 画像は取りこぼしても構わない (最新フレームだけ見る)。odom は
        # 欠けると経路追従が破綻するので信頼性を落とさない。
        img_qos = QoSProfile(depth=1,
                             reliability=QoSReliabilityPolicy.BEST_EFFORT)

        self.cmd_pub = self.node.create_publisher(
            Twist, f"{robot_ns}/cmd_vel", 10)
        self.node.create_subscription(
            Odometry, f"{robot_ns}/odom", self._odom_callback, 10)
        self.node.create_subscription(
            Image, f"{robot_ns}/image_raw2", self._image_callback, img_qos)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        self.current_pose = None
        self.current_image = None

        self.cross_lines = self._load_cross_lines(_resolve_data_file(
            "JEPA_CROSS_MARKERS", "cross_markers_400.csv",
            "通過判定線の CSV (列 x1, y1, x2, y2)"))
        
        self.visited_lines = set()
        self.total_lines = len(self.cross_lines)
        self.use_proximity_mode = True  # default strict crossing

        self.action_space = spaces.Discrete(5)

        # Observation: RGB camera image normalized to (3, 84, 84)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(3, 84, 84), dtype=np.float32
        )  

        print(self.total_lines)

    def _load_cross_lines(self, filepath):
        lines = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                x1, y1, x2, y2 = float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2'])
                lines.append(((x1, y1), (x2, y2)))
        return lines
    

    def set_visual_mode(self, mode: int):
        self.visual_mode = int(mode)



    def _check_crossed_lines(self, prev_pos, curr_pos):
        """
        Returns indices of newly crossed lines since last step.
        """
        new_crossed = []
        for i, ((x1, y1), (x2, y2)) in enumerate(self.cross_lines):
            if i in self.visited_lines:
                continue

            # Vector cross product check for line intersection
            def ccw(A, B, C):
                return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
            A, B = prev_pos, curr_pos
            C, D = (x1, y1), (x2, y2)
            intersect = ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
            
            if intersect:
                new_crossed.append(i)

        return new_crossed

    def add_random_grass_patches(self,img, patch_size=40, patch_count=2):
        """
        Add randomized grass-colored square patches to the background,
        mimicking Gym CarRacing's textured field.
        """
        h, w, _ = img.shape
        for _ in range(patch_count):
            # Random top-left position
            x = random.randint(0, w - patch_size)
            y = random.randint(0, h - patch_size)

            # Randomized green variation
            green_variation = np.random.randint(-10, 20)
            patch_color = (
                102,
                230,
                102,
            )  # BGR (grass patch accent)

            # Draw patch
            cv2.rectangle(
                img,
                (x, y),
                (x + patch_size, y + patch_size),
                patch_color,
                thickness=-1,
            )



    def draw_gym_car_shape(self,image, center=(320, 410), scale=0.25):
        """
        Draw a Gym CarRacing-style car with correct body and wheels.
        Args:
        image: target BGR image
        center: (x, y) center of car in image pixels
        scale: scaling factor
        """

        # ===== Original car polygons =====
        HULL_POLY1 = [(-60, +130), (+60, +130), (+60, +110), (-60, +110)]
        HULL_POLY2 = [(-15, +120), (+15, +120), (+20, +20), (-20, +20)]
        HULL_POLY3 = [
            (+25, +20),
            (+50, -10),
            (+50, -40),
            (+20, -90),
            (-20, -90),
            (-50, -40),
            (-50, -10),
            (-25, +20),
        ]
        HULL_POLY4 = [(-50, -120), (+50, -120), (+50, -90), (-50, -90)]

        polys = [HULL_POLY1, HULL_POLY2, HULL_POLY3, HULL_POLY4]

        # ===== Original wheel geometry =====
        WHEELPOS = [(-55, +80), (+55, +80), (-55, -82), (+55, -82)]
        WHEEL_R = 27
        WHEEL_W = 14

        # ===== Colors from the real Gym car =====
        HULL_COLOR = (0, 0, 202)     # deep red (BGR)
        WHEEL_COLOR = (0, 0, 0)      # black
        WHEEL_WHITE = (77, 77, 77)   # gray/white overlay

        cx, cy = center

        # ===== Draw car body =====
        for poly in polys:
            pts = np.array([[int(cx + x * scale), int(cy - y * scale)] for x, y in poly], np.int32)
            cv2.fillPoly(image, [pts], HULL_COLOR)

        # ===== Draw 4 tires =====
        for wx, wy in WHEELPOS:
            w = int(WHEEL_W * scale)
            h = int(WHEEL_R * 2 * scale)
            x = int(cx + wx * scale)
            y = int(cy - wy * scale)

            top_left = (x - w, y - h // 2)
            bottom_right = (x + w, y + h // 2)

            # Tire base
            cv2.rectangle(image, top_left, bottom_right, WHEEL_COLOR, thickness=-1)

            # Small white highlight stripe
            highlight_y1 = int(y - h * 0.3)
            highlight_y2 = int(y - h * 0.15)
            cv2.rectangle(image, (x - w, highlight_y1), (x + w, highlight_y2), WHEEL_WHITE, thickness=-1)
    

    def transform_image_cv2_dark(self, img, min_road_width=15, close_kernel=9, open_kernel=5, add_grass=False):
        """
        Clean Gazebo camera frame → stylized Gym-like view:
        - Road → gray (102,102,102)
        - Background → green (102,204,102)
        - Removes trees, shadows, barriers using color filtering + morphology.
        - Draws a simple car body overlay.
        """

        if img is None:
            raise FileNotFoundError("Empty frame in transform_image_cv2")

        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # --- Step 1: Detect dark/gray road --- [0, 6, 0] [180, 81, 80]
        lower_road = np.array([0, 6, 0])
        upper_road = np.array([180, 81, 80])
        mask_road = cv2.inRange(hsv, lower_road, upper_road)

        # --- Step 2: Suppress green & sky ---  
        lower_green = np.array([30, 60, 40]) 
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_road[mask_green > 0] = 0  # remove green areas from road mask

        # --- Step 3: Morphological cleaning ---
        close_k = np.ones((close_kernel, close_kernel), np.uint8)
        open_k = np.ones((open_kernel, open_kernel), np.uint8)
        mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_CLOSE, close_k)
        mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_OPEN, open_k)

        # --- Step 4: Strengthen the road area ---
        mask_road = cv2.dilate(mask_road, np.ones((min_road_width, min_road_width), np.uint8))
        road_pixel_count = cv2.countNonZero(mask_road)
        # print("Dark pixel:",road_pixel_count)
        if road_pixel_count < 60:
            h_img, w_img = mask_road.shape
            left_width = int(0.25 * w_img)  # left 25% of image

            fallback_mask = np.zeros_like(mask_road)
            fallback_mask[int(0.5 * h_img):, :left_width] = 255  # bottom-left region

            mask_road = cv2.bitwise_or(mask_road, fallback_mask)

        # --- Step 5: Compose clean output ---
        new_img = np.full_like(img, (100, 202, 100))  # background green

        if add_grass:
            self.add_random_grass_patches(new_img, patch_size=150, patch_count=3)

        # Paint road gray
        new_img[mask_road > 0] = (102, 102, 102)

        

        # --- Step 6: Add car representation ---
        self.draw_gym_car_shape(new_img, center=(320, 410), scale=0.25)

        return new_img

    def transform_image_cv2_blue(self, img, min_road_width=15, close_kernel=9, open_kernel=5, add_grass=False):
        """
        Clean Gazebo camera frame → stylized Gym-like view:
        - Road → gray (102,102,102)
        - Background → green (102,204,102)
        - Removes trees, shadows, barriers using color filtering + morphology.
        - Draws a simple car body overlay.
        """

        if img is None:
            raise FileNotFoundError("Empty frame in transform_image_cv2")

        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # --- Step 1: Detect dark/gray road ---
        lower_road = np.array([0, 125, 0]) #[119, 0, 63]
        upper_road = np.array([180, 255, 95])  #[180, 226, 177]
        mask_road = cv2.inRange(hsv, lower_road, upper_road)

        # --- Step 2: Suppress green & sky ---
        lower_green = np.array([30, 60, 40])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_road[mask_green > 0] = 0  # remove green areas from road mask

        # --- Step 3: Morphological cleaning ---
        close_k = np.ones((close_kernel, close_kernel), np.uint8)
        open_k = np.ones((open_kernel, open_kernel), np.uint8)
        mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_CLOSE, close_k)
        mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_OPEN, open_k)

        # --- Step 4: Strengthen the road area ---
        mask_road = cv2.dilate(mask_road, np.ones((min_road_width, min_road_width), np.uint8))
        road_pixel_count = cv2.countNonZero(mask_road)

        if road_pixel_count < 60:
            h_img, w_img = mask_road.shape
            left_width = int(0.25 * w_img)  # left 25% of image

            fallback_mask = np.zeros_like(mask_road)
            fallback_mask[int(0.5 * h_img):, :left_width] = 255  # bottom-left region

            mask_road = cv2.bitwise_or(mask_road, fallback_mask)

        # --- Step 5: Compose clean output ---
        new_img = np.full_like(img, (100, 202, 100))  # background green

        if add_grass:
            self.add_random_grass_patches(new_img, patch_size=150, patch_count=3)

        # Paint road gray
        new_img[mask_road > 30] = (102, 102, 102)
        # if(mask_road<20):
        #     new_img = np.full_like(img, (100, 202, 100))

        # --- Step 6: Add car representation ---
        self.draw_gym_car_shape(new_img, center=(320, 410), scale=0.25)

        return new_img


    # def patched_transform(self,img):

    #     img = cv2.resize(img,(640,480))
    #     img =  self.transform_image_cv2(img)
    
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img = cv2.resize(img, (84, 84))
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     img = img.astype(np.float32) / 255.0
    #     return img
    

    def patched_transform(self,img):
        """
        Accepts:
        - torch.Tensor [3,H,W] or [H,W] or [H,W,3]
        - numpy array with same shapes
        Returns:
        - grayscale float32 [84,84] in [0,1]
        """
        # ---- torch -> numpy ----
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        # ---- handle CHW -> HWC ----
        if img.ndim == 3 and img.shape[0] in (1, 3):          # CHW
            img = np.transpose(img, (1, 2, 0))                # HWC

        # ---- if grayscale HxW -> make it 3ch BGR ----
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # ---- if float [0,1] -> uint8 ----
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)

        # now img is HWC uint8 (assumed BGR)
        img = cv2.resize(img, (640, 480))
        # cv2.imshow("resized",img)
        img = self.transform_image_cv2_2(img)   # expects BGR HWC

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (84, 84))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = img.transpose(1, 2, 0)
        img = cv2.resize(img, (84, 84))
        # cv2.imshow("gym_img",img)
        # cv2.waitKey(1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / 255.0

        return img
    
    def patched_transform2(self,img):
        """
        Accepts:
        - torch.Tensor [3,H,W] or [H,W] or [H,W,3]
        - numpy array with same shapes
        Returns:
        - grayscale float32 [84,84] in [0,1]
        """
        # ---- torch -> numpy ----
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        # ---- handle CHW -> HWC ----
        if img.ndim == 3 and img.shape[0] in (1, 3):          # CHW
            img = np.transpose(img, (1, 2, 0))                # HWC

        # ---- if grayscale HxW -> make it 3ch BGR ----
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # ---- if float [0,1] -> uint8 ----
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)

        # now img is HWC uint8 (assumed BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))
        # cv2.imwrite("saved_image.jpg", img)

        # cv2.imshow("resized",img)
        img = self.transform_image_cv2_2(img)   # expects BGR HWC

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (84, 84))
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))  # CHW -> HWC

        # Resize (safe for both)
        img = cv2.resize(img, (84, 84))
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img = img.astype(np.float32) / 255.0


        # img = img.astype(np.float32) / 255.0
        # img = np.transpose(img, (2, 0, 1))
        # img = img.transpose(1, 2, 0)
        # img = cv2.resize(img, (84, 84))
        # # cv2.imshow("gym_img",img)
        # # cv2.waitKey(1)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = img.astype(np.float32) / 255.0

        return img


    def transform_image_cv2(self, img, min_road_width=15, close_kernel=9, open_kernel=5, add_grass=False):
        """
        Clean Gazebo camera frame → stylized Gym-like view:
        - Road → gray (102,102,102)
        - Background → green (102,204,102)
        - Removes trees, shadows, barriers using color filtering + morphology.
        - Draws a simple car body overlay.
        """

        if img is None:
            raise FileNotFoundError("Empty frame in transform_image_cv2")

        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # --- Step 1: Detect dark/gray road ---
        # lower_road = np.array([0, 0, 20])
        # upper_road = np.array([180, 161, 183])
        # lower_road = np.array([0, 0, 0])
        # upper_road = np.array([80, 255, 125])
        lower_road = np.array([0, 0, 40])
        upper_road = np.array([180, 90, 140])
        mask_road = cv2.inRange(hsv, lower_road, upper_road)

        # --- Step 2: Suppress green & sky ---
        lower_green = np.array([30, 60, 40])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_road[mask_green > 0] = 0  # remove green areas from road mask

        # --- Step 3: Morphological cleaning ---
        close_k = np.ones((close_kernel, close_kernel), np.uint8)
        open_k = np.ones((open_kernel, open_kernel), np.uint8)
        mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_CLOSE, close_k)
        mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_OPEN, open_k)

        # --- Step 4: Strengthen the road area ---
        mask_road = cv2.dilate(mask_road, np.ones((min_road_width, min_road_width), np.uint8))
        road_pixel_count = cv2.countNonZero(mask_road)
        # print("White pixel",road_pixel_count)
        # --- Step 5: Compose clean output ---
        new_img = np.full_like(img, (100, 202, 100))  # background green

        if add_grass:
            self.add_random_grass_patches(new_img, patch_size=150, patch_count=3)

        # Paint road gray
        new_img[mask_road > 0] = (102, 102, 102)

        # --- Step 6: Add car representation ---
        self.draw_gym_car_shape(new_img, center=(320, 410), scale=0.25)

        return new_img


    def transform_image_cv2_2(self, img, min_road_width=13, close_kernel=9, open_kernel=5, add_grass=False):
        """
        Clean Gazebo camera frame → stylized Gym-like view:
        - Road → gray (102,102,102)
        - Background → green (102,204,102)
        - Removes trees, shadows, barriers using color filtering + morphology.
        - Draws a simple car body overlay.
        """

        if img is None:
            raise FileNotFoundError("Empty frame in transform_image_cv2")

        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # --- Step 1: Detect dark/gray road ---
        lower_road = np.array([0, 0, 40])
        upper_road = np.array([180,255,158])
        # lower_road = np.array([0, 0, 44])
        # upper_road = np.array([180, 57, 220])
        mask_road = cv2.inRange(hsv, lower_road, upper_road)

        # --- Step 2: Suppress green & sky ---
        lower_green = np.array([30, 60, 40])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_road[mask_green > 0] = 0  # remove green areas from road mask

        # --- Step 3: Morphological cleaning ---
        close_k = np.ones((close_kernel, close_kernel), np.uint8)
        open_k = np.ones((open_kernel, open_kernel), np.uint8)
        mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_CLOSE, close_k)
        mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_OPEN, open_k)

        # --- Step 4: Strengthen the road area ---
        mask_road = cv2.dilate(mask_road, np.ones((min_road_width, min_road_width), np.uint8))
        road_pixel_count = cv2.countNonZero(mask_road)
        # print("White pixel",road_pixel_count)
        # --- Step 5: Compose clean output ---
        new_img = np.full_like(img, (100, 202, 100))  # background green

        if add_grass:
            self.add_random_grass_patches(new_img, patch_size=150, patch_count=3)

        # Paint road gray
        new_img[mask_road > 0] = (102, 102, 102)

        # --- Step 6: Add car representation ---
        self.draw_gym_car_shape(new_img, center=(320, 410), scale=0.25)

        return new_img
    
    # def transform_image_cv2(
    #     self,
    #     img,
    #     road_width_px=150,
    #     add_grass=False
    # ):
    #     """
    #     Gazebo camera frame → Gym-style road image

    #     - ONLY road is shown
    #     - Road width is CONSTANT
    #     - Road edges are SMOOTH & STRAIGHT
    #     - Background is uniform green
    #     - ROS callback safe
    #     """

    #     if img is None:
    #         raise FileNotFoundError("Empty frame in transform_image_cv2")

    #     h, w, _ = img.shape

    #     # ===============================
    #     # 1. Rough road detection (HSV)
    #     # ===============================
    #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #     lower_road = np.array([0, 0, 40])
    #     upper_road = np.array([180, 90, 140])
    #     mask = cv2.inRange(hsv, lower_road, upper_road)

    #     # Remove green areas
    #     lower_green = np.array([35, 60, 40])
    #     upper_green = np.array([90, 255, 255])
    #     green_mask = cv2.inRange(hsv, lower_green, upper_green)
    #     mask[green_mask > 0] = 0

    #     # ===============================
    #     # 2. Keep largest vertical component
    #     # ===============================
    #     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    #         mask, connectivity=8
    #     )

    #     road_label = -1
    #     best_score = 0

    #     for i in range(1, num_labels):
    #         area = stats[i, cv2.CC_STAT_AREA]
    #         height = stats[i, cv2.CC_STAT_HEIGHT]

    #         # Road must be tall and large
    #         if height > h * 0.3:
    #             score = area * height
    #             if score > best_score:
    #                 best_score = score
    #                 road_label = i

    #     if road_label < 0:
    #         return np.full_like(img, (102, 204, 102))

    #     road_mask = (labels == road_label).astype(np.uint8) * 255

    #     # ===============================
    #     # 3. Compute centerline per row
    #     # ===============================
    #     center_x = np.zeros(h, dtype=np.int32)
    #     valid = np.zeros(h, dtype=bool)

    #     for y in range(h):
    #         xs = np.where(road_mask[y] > 0)[0]
    #         if len(xs) > 5:
    #             center_x[y] = int(xs.mean())
    #             valid[y] = True

    #     # Interpolate missing rows
    #     idx = np.where(valid)[0]
    #     if len(idx) < 10:
    #         return np.full_like(img, (102, 204, 102))

    #     center_x = np.interp(
    #         np.arange(h),
    #         idx,
    #         center_x[idx]
    #     ).astype(np.int32)

    #     # ===============================
    #     # 4. Smooth centerline (CRITICAL)
    #     # ===============================
    #     center_x = cv2.GaussianBlur(
    #         center_x.reshape(-1, 1).astype(np.float32),
    #         (1, 51),
    #         0
    #     ).reshape(-1).astype(np.int32)

    #     # ===============================
    #     # 5. Rebuild road with fixed width
    #     # ===============================
    #     road_half = road_width_px // 2
    #     new_img = np.full_like(img, (102, 204, 102))

    #     for y in range(h):
    #         cx = int(center_x[y])
    #         x1 = max(0, cx - road_half)
    #         x2 = min(w - 1, cx + road_half)
    #         new_img[y, x1:x2] = (102, 102, 102)

    #     # ===============================
    #     # 6. Optional grass noise
    #     # ===============================
    #     if add_grass:
    #         self.add_random_grass_patches(new_img, 150, 3)

    #     # ===============================
    #     # 7. Draw Gym-style car
    #     # ===============================
    #     self.draw_gym_car_shape(
    #         new_img,
    #         center=(w // 2, int(h * 0.85)),
    #         scale=0.25
    #     )

    #     return new_img




    def _action_to_cmd_cont(self,action):
        return action[0], action[1]

    # def _action_to_cmd(self, action,
    #                dt=0.02,
    #                max_linear_speed=0.9,
    #                max_angular_speed=0.8,
    #                accel_rate=12,
    #                brake_rate=5.0,
    #                coast_deceleration=1.2,
    #                steer_rate=9):
    #     """Discrete Gym-style control → (linear_x, angular_z)"""

    #     # Decode one-hot or index action
    #     if isinstance(action, (list, np.ndarray)):
    #         a = np.array(action).flatten()
    #         action_idx = int(np.argmax(a)) if not np.all(a == 0) else 0
    #     else:
    #         action_idx = int(action)

    #     # Initialize control flags
    #     apply_gas = False
    #     apply_brake = False
    #     desired_steer = 0.0

    #     # Action logic
    #     if action_idx == 1:   # steer right
    #         desired_steer = -1.0
    #     elif action_idx == 2: # steer left
    #         desired_steer = +1.0
    #     elif action_idx == 3: # gas
    #         apply_gas = True
    #     elif action_idx == 4: # brake
    #         apply_brake = True
    #     # else 0 = coast

    #     # --- Smooth steering ---
    #     steer_delta = desired_steer - getattr(self, "_current_steer", 0.0)
    #     max_steer_step = steer_rate * dt
    #     if abs(steer_delta) <= max_steer_step:
    #         self._current_steer = desired_steer
    #     else:
    #         self._current_steer += np.sign(steer_delta) * max_steer_step
    #     self._current_steer = float(np.clip(self._current_steer, -1.0, 1.0))

    #     # --- Smooth speed ---
    #     if apply_gas:
    #         self._current_speed += accel_rate * dt
    #     elif apply_brake:
    #         self._current_speed -= brake_rate * dt
    #     elif action_idx == 0:  # coast
    #         if self._current_speed > 0:
    #             self._current_speed -= coast_deceleration * dt

    #     self._current_speed = float(np.clip(self._current_speed, 0.0, max_linear_speed))

    #     # --- Convert to angular velocity ---
    #     angular_z = float(self._current_steer * max_angular_speed)

    #     # Prevent oscillation when almost stopped
    #     if self._current_speed < 0.05 or action_idx == 4 or action_idx == 0 or action_idx == 3:
    #         angular_z = 0

    #     if(self._current_speed<0.05):
    #         self.brake_frames +=1

    #     else:
    #         self.brake_frames =0

    #     return self._current_speed, angular_z
    
    # def _action_to_cmd(self, action,
    #                dt=0.02,
    #                max_linear_speed=0.9,
    #                max_angular_speed=0.8,
    #                accel_rate=12,
    #                brake_rate=5.0,
    #                coast_deceleration=1.2,
    #                steer_rate=9):

    def _action_to_cmd(self, action,
                   dt=0.02,
                   max_linear_speed=1.0, #default 1.0
                   max_angular_speed=1,
                   accel_rate=10,
                   brake_rate=3,
                   coast_deceleration=1.2,
                   steer_rate=10):
        """Discrete Gym-style control → (linear_x, angular_z)"""

        # Decode one-hot or index action
        if isinstance(action, (list, np.ndarray)):
            a = np.array(action).flatten()
            action_idx = int(np.argmax(a)) if not np.all(a == 0) else 0
        else:
            action_idx = int(action)

        # Initialize control flags
        apply_gas = False
        apply_brake = False
        desired_steer = 0.0

        # Action logic
        if action_idx == 1:   # steer right
            desired_steer = -1.0
        elif action_idx == 2: # steer left
            desired_steer = +1.0
        elif action_idx == 3: # gas
            apply_gas = True
        elif action_idx == 4: # brake
            apply_brake = True
        # else 0 = coast

        # --- Smooth steering ---
        steer_delta = desired_steer - getattr(self, "_current_steer", 0.0)
        max_steer_step = steer_rate * dt
        if abs(steer_delta) <= max_steer_step:
            self._current_steer = desired_steer
        else:
            self._current_steer += np.sign(steer_delta) * max_steer_step
        self._current_steer = float(np.clip(self._current_steer, -1.0, 1.0))

        # --- Smooth speed ---
        if apply_gas:
            self._current_speed += accel_rate * dt
        elif apply_brake:
            self._current_speed -= brake_rate * dt
        elif action_idx == 0:  # coast
            if self._current_speed > 0:
                self._current_speed -= coast_deceleration * dt

        self._current_speed = float(np.clip(self._current_speed, 0.0, max_linear_speed))

        # --- Convert to angular velocity ---
        angular_z = float(self._current_steer * max_angular_speed)

        # Prevent oscillation when almost stopped
        if self._current_speed < 0.05 or action_idx == 4 or action_idx == 0 or action_idx == 3:
            angular_z = 0

        if(self._current_speed<0.005):
            self.brake_frames +=1
            print(self.brake_frames)

        else:
            self.brake_frames =0

        return self._current_speed, angular_z




    def _find_nearest_point(self,pose):
        """Return nearest point, tangent yaw, and shortest distance."""
        path_xy = [(p[0], p[1]) for p in self.path]
        px = pose[0]
        py = pose[1]

        def closest_point_on_segment(px, py, ax, ay, bx, by):
            vx = bx - ax
            vy = by - ay
            wx = px - ax
            wy = py - ay
            denom = vx*vx + vy*vy
            if denom == 0.0:
                return (ax, ay, 0.0)
            t = (wx*vx + wy*vy) / denom
            t = max(0.0, min(1.0, t))
            cx = ax + t * vx
            cy = ay + t * vy
            return (cx, cy, t)


        best_d = float('inf')
        best_point = None
        best_seg = 0
        for i in range(len(path_xy) - 1):
            ax, ay = path_xy[i]
            bx, by = path_xy[i + 1]
            cx, cy, _ = closest_point_on_segment(px, py, ax, ay, bx, by)
            d = math.hypot(px - cx, py - cy)
            if d < best_d:
                best_d = d
                best_point = (cx, cy)
                best_seg = i

        # tangent direction
        ax, ay = path_xy[best_seg]
        bx, by = path_xy[best_seg + 1]
        path_yaw = math.atan2(by - ay, bx - ax)

        return best_point ,best_d  #, path_yaw, best_d


    def _odom_callback(self,msg):
        self.current_pose = msg.pose.pose

    def _image_callback(self,msg):
        self.img_src= imgmsg_to_bgr8(msg)
        cleaned = cv2.GaussianBlur(self.img_src, (3,3), 0)

        self.current_image = cleaned #self.transform_image_cv2_dark(cleaned)  #(self.img_src.copy())


    def _load_path(self,path_file):
        path = []
        with open(path_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                path.append((float(row['x']), float(row['y']), float(row['yaw'])))
                # rospy.loginfo(f"Loaded {len(path)} trajectory points from {path_file}")
        return path

    def _get_yaw_from_quat(self,q):
        """Extract yaw from quaternion."""
        _, _, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        return yaw


    def _distance(self,x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)



    def get_obs_with_mode(self, mode: int):
        """
        Return Dreamer-ready observation for a given visual mode.
        """

        file_name = "img_"+str(mode)+".jpg"
        if self.current_image is None:
            return None

        if mode == 0:
            img = self.transform_image_cv2(self.current_image)
            
        elif mode == 1:
            img = self.transform_image_cv2_dark(self.current_image)

        elif mode == 2:
            img = self.transform_image_cv2_blue(self.current_image)

        else:
            raise ValueError("Invalid visual mode")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (84, 84))
        cv2.imwrite(file_name, img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / 255.0
        # img = np.transpose(img, (2, 0, 1))  # CHW

        return img
    
    # def add_mud_patch(self,image):

    #     if isinstance(image, torch.Tensor):
    #         img = (image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    #     else:
    #         img = image.copy()
    #         if img.dtype != np.uint8:
    #             img = (img * 255).astype("uint8")

    #     h, w, _ = img.shape
    #     overlay = img.copy()

    #     num_patches = random.randint(5, 7)
    #     for _ in range(num_patches):
    #         cx = random.randint(0, w - 1)
    #         cy = random.randint(0, h - 1)
    #         r  = random.randint(10, 20)
    #         cv2.circle(overlay, (cx, cy), r, MUD_COLOR, -1)

    #     alpha = 1.0  # same as training
    #     muddy = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    #     muddy = torch.from_numpy(muddy).permute(2, 0, 1).float() / 255.0
    #     return muddy
    
    def add_mud_patch(self,image):
        """
        image: torch.Tensor [3,84,84] or numpy [84,84,3]
        returns: torch.Tensor [3,84,84]
        """

        # ---- to numpy HWC uint8 ----
        if isinstance(image, torch.Tensor):
            img = (image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        else:
            img = image.copy()
            if img.dtype != np.uint8:
                img = (img * 255).astype("uint8")

        h, w, _ = img.shape
        overlay = img.copy()

        num_patches = random.randint(5, 7)
        for _ in range(num_patches):
            cx = random.randint(0, w - 1)
            cy = random.randint(0, h - 1)
            r  = random.randint(10, 15)
            cv2.circle(overlay, (cx, cy), r, MUD_COLOR, -1)

        alpha = 1.0  # hard mud (can change to 0.6 if needed)
        muddy = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # ---- back to torch CHW float ----
        muddy = torch.from_numpy(muddy).permute(2, 0, 1).float() / 255.0
        return muddy


    def get_obs_with_patch(self, mode: int,patch:bool):
        

        file_name = "img_"+str(mode)+".jpg"
        if self.current_image is None:
            return None

        if patch == False:
            img = self.transform_image_cv2(self.current_image)
            src_img = cv2.resize(self.current_image, (84, 84))
            
        
        else:
            input = cv2.resize(self.current_image, (84, 84))
            patched = self.add_mud_patch(input)
            src_img = (
                patched.permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            src_img = (src_img * 255).astype(np.uint8)

            img = self.transform_image_cv2(src_img)
            

        # else:
        #     raise ValueError("Invalid visual mode")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (84, 84))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / 255.0

        

        return src_img,img


    def _get_obs(self):
        """Return latest camera image as normalized (3,84,84)."""

        # if self.visual_mode == 5:


        if self.visual_mode == 0:
            img_src = self.transform_image_cv2(self.current_image)

        elif self.visual_mode == 1:
            img_src = self.transform_image_cv2_dark(self.current_image)

        elif self.visual_mode == 2:
            img_src = self.transform_image_cv2_blue(self.current_image)

        elif self.visual_mode == 3:
            patched_img = self.add_mud_patch(self.current_image)
            # cv2.imshow("patched_img", patched_img)
            # cv2.waitKey(1)

            img_src = self.transform_image_cv2(patched_img)

        # img_src = self.transform_image_cv2_dark(self.current_image)
        img = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (84, 84))
        # cv2.imshow("GazeboEnv Camera", img)
        # cv2.waitKey(1)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return img

    def reset(self, seed=None, options=None):

        # Classic は /gazebo/get_physics_properties で一時停止状態を問い合わせて
        # いたが、Harmonic に同等のサービスはない。状態を読まずに常に unpause を
        # 送る方が単純で、既に走っていても無害である。
        self.gz.unpause()

        """Reset the environment and return the first observation."""
        # super().reset(seed=seed)
        self.node.get_logger().info("Resetting GazeboEnv...")
        self._stop_robot()

        self.current_pose = None
        self.current_image = None
        self.target_index = 0
        self.done = False
        self.total_reward = 0.0
        self.episode_start_time = time.time()
        self.visited_lines.clear()
        self.prev_pos = None
        self.prev_closest = None
        self.done = False
        self.terminated = False
        self.too_long = False
        self.min_rewad_terminated = False

        self._current_speed = 0.0
        self._current_steer = 0.0
        self._last_time = None
        self._last_desired_steer = 0.0

        self.last_move_time = time.time()
        self.last_position = None
        self.flag_no_obs = True

        self.img_src = None
        self.brake_frames = 0
        self.visual_mode = 0
       

        
        # ロボットを初期姿勢へ戻す。Classic の /gazebo/set_model_state を
        # Harmonic の /world/<name>/set_pose に置き換えた。
        #
        # Classic の SetModelState は twist (速度) も 0 に指定できたが、
        # Harmonic の set_pose は姿勢のみである。速度を止めるため cmd_vel に 0 を
        # 送ってから姿勢を変える。
        #
        # reset_all() は使えない。world SDF に書かれていない実行時 spawn の
        # モデルを削除してしまい、launch で置いたロボットが world から消える
        # (gz_world_control.py の docstring 参照)。
        #
        # 初期姿勢は jetbot_tellus.launch.py の既定値と揃えている。
        x_pos = float(os.environ.get("JEPA_INIT_X", 5.163443))
        y_pos = float(os.environ.get("JEPA_INIT_Y", 7.766847))
        z_pos = float(os.environ.get("JEPA_INIT_Z", 0.0))
        yaw = float(os.environ.get("JEPA_INIT_YAW", -1.573762))

        self.cmd_pub.publish(Twist())  # 速度を 0 にする

        try:
            self.gz.set_pose(self.robot_model_name,
                             x=x_pos, y=y_pos, z=z_pos, yaw=yaw)
            self.node.get_logger().info("Robot reset to initial position.")
        except Exception as e:
            self.node.get_logger().error(f"Failed to reset robot model: {e}")

        # Wait for sensors to update
        # rospy.sleep(1.0)
        # センサの初回データを待つ。ここは sim 時間で待つ意味がない
        # (コールバックが届くかどうかだけの問題) ので壁時計で待つ。
        # sim 時間で待つと実時間係数が低い環境で _sleep の安全弁に当たり、
        # 「完了しなかった」警告が出てしまう。
        _wait_start = time.monotonic()
        while (self.current_pose is None or self.current_image is None) and self._ok():
            if time.monotonic() - _wait_start > 30.0:
                self.node.get_logger().error(
                    "30 秒待っても odom または image_raw2 が届かない。"
                    "Gazebo と ros_gz_bridge が起動しているか、"
                    f"トピック名 ({self.robot_ns}/odom, "
                    f"{self.robot_ns}/image_raw2) が合っているか確認すること。")
                break
            time.sleep(0.05)

        # Get observation
        # rospy.sleep(0.1)
        obs = self._get_obs()
        info = {}
        return obs, info

    def _check_fall(self):
        """Detect if the robot has fallen or tilted excessively."""
        if self.current_pose is None:
            return False

        # Extract quaternion
        q = self.current_pose.orientation
        roll, pitch, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])

        # Check if roll or pitch exceeds threshold (radians)
        if abs(roll) > 0.5 or abs(pitch) > 0.5:  # ~30 degrees tilt
            return True
        return False


    def step(self, action):

        # Classic は毎 step で /gazebo/get_physics_properties を呼んで一時停止
        # 状態を確認し、止まっていれば unpause していた。Harmonic に状態取得の
        # サービスはないので、常に unpause を送る。
        #
        # なお毎 step サービスを叩くのは無駄なので、本来は reset の時だけで
        # 足りる。元の実装を踏襲して残しているが、学習速度が問題になるなら
        # ここは削れる。
        self.gz.unpause()

        """Apply action and return (obs, reward, done, truncated, info)."""
        if self.current_pose is None or self.current_image is None:
            self.node.get_logger().warn("No data yet.")
            # return np.zeros(self.observation_space.shape), 0.0, False, False, {}
        
        
        # action = np.array(action).flatten()

        # # If one-hot → convert to integer index
        # if action.ndim > 0:
        #     action_idx = int(np.argmax(action))
        # else:
        #     action_idx = int(action)

        linear_speed,angular_z= self._action_to_cmd(action)
        # linear_speed,angular_z = self._action_to_cmd_cont(action)

        # --- Map discrete index to angular velocity ---
        # angular_z_values = [0.0, 0.3, 0.6, -0.3, -0.6]
        # angular_z = angular_z_values[action_idx]

        # rospy.loginfo(f"linear_speed: {linear_speed} → angular_z={angular_z:.2f}")
        cmd = Twist()
        # rclpy の float フィールドは Python の float を厳格に要求する
        # (PyFloat_Check)。rospy は int や numpy スカラを暗黙に受け付けていたが、
        # ROS 2 では次のように落ちる。
        #   geometry_msgs__msg__vector3__convert_from_py:
        #   Assertion `PyFloat_Check(field)' failed.
        # gymnasium の action は numpy スカラで渡ってくるため、明示的に変換する。
        cmd.linear.x = float(linear_speed)
        cmd.angular.z = float(angular_z)
        self.cmd_pub.publish(cmd)

        # time.sleep(1) 
        self._sleep(0.0350)
        # pause_gazebo()

        obs = self._get_obs()
        reward, terminated ,truncated = self._compute_reward()
        # 元は毎 step の最後に pause_gazebo() を呼んでいたが、直後の
        # step() 冒頭で unpause しており実質無意味なうえ、サービス往復が
        # 2 倍になる。観測は上の _get_obs() で取り終えているので外した。
        # print(obs.shape)
        info = {}
        # if self.render_mode == "human":
            # cv2.imshow("GazeboEnv Camera",cv2.resize(self.img_src, (84, 84)))
            # cv2.imshow("current_image",self.current_image)
            # cv2.waitKey(1)
        # print(f"gazebo env rew : {reward}")
        return obs, reward, terminated, truncated , info  #, time_terminated ,min_rewad_terminated   #, truncated, info

    def _compute_reward(self):
        if self.current_pose is None:
            return 0.0,False, False

        rx = self.current_pose.position.x
        ry = self.current_pose.position.y
        curr_pos = (rx, ry)

        if not hasattr(self, "episode_start_time"):
            self.episode_start_time = time.time()
            self.total_reward = 0.0

        elapsed_time = time.time() - self.episode_start_time

    # --- Time limit termination ---
        too_long = False
        if elapsed_time > 3600:  # 1 hour = 3600 seconds
            self.node.get_logger().warn("Episode terminated due to time limit (1 hour).")
            too_long = True

        # If first step, skip crossing check
        if not hasattr(self, "prev_pos"):
            self.prev_pos = curr_pos
            self.prev_closest = curr_pos
            return -0.1, False,False

        curr_closest , shortest_dist = self._find_nearest_point(curr_pos)
        off_path = shortest_dist > self.max_deviation

        new_lines = []
        if not self.use_proximity_mode:
            if self.prev_pos is not None and curr_pos is not None:
                
                new_lines = self._check_crossed_lines(self.prev_pos, curr_pos)

        else:
            if self.prev_closest is not None and curr_closest is not None:
                new_lines = self._check_crossed_lines(self.prev_closest, curr_closest)

        # Check crossed lines
        # new_lines = self._check_crossed_lines(self.prev_pos, curr_pos)
        for i in new_lines:
            self.visited_lines.add(i)

        crossed_count = len(new_lines)
        # if(crossed_count>0):
        #     print(crossed_count)

        self.prev_pos = curr_pos
        self.prev_closest = curr_closest
        # print(len(self.visited_lines))
        # === Gym CarRacing-style reward ===
        frame_penalty = -0.1
        progress_reward = (1000.0 / self.total_lines) * crossed_count
        reward = frame_penalty + progress_reward

        # Termination when all lines visited or robot finished
        terminated = len(self.visited_lines) >= (self.total_lines-32) #-32
        # print("Visited lines:", len(self.visited_lines)) 
        self.total_reward += reward

        # rospy.loginfo(self.total_reward)
        fall_down = self._check_fall()

        min_rewad_terminated = False
        if self.total_reward <= -300.0:
            min_rewad_terminated = True

        brake_stop_terminated = False
        if getattr(self, "brake_frames", 0) >= getattr(self, "brake_terminate_threshold", 10):
            self.node.get_logger().warn("Robot braked and stopped for too long, terminating episode.")
            brake_stop_terminated = True

        min_rewad_terminated = False
        if self.total_reward <= -300.0:
            min_rewad_terminated = True

        truncated =  fall_down or min_rewad_terminated or  off_path or brake_stop_terminated #or  off_path   #or too_long or  off_path 
        # print(off_path,fall_down,min_rewad_terminated,brake_stop_terminated,too_long)
        # print("total_reward: " , self.total_reward , "min_rewad_terminated: ",min_rewad_terminated)
        if(truncated):
            reward = -100.0
        
        return reward, terminated ,truncated    # ,time_terminated,min_rewad_terminated

    
    def _stop_robot(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)
        if self.render_mode == "human":
            cv2.destroyAllWindows()
        self.node.get_logger().info("Robot stopped.")

    # ---- rospy.sleep / rospy.is_shutdown の代替 -------------------------

    def _sleep(self, seconds, wall_timeout=None):
        """sim 時間で待つ。rospy.sleep の代替。

        rospy.sleep は /use_sim_time が有効なとき sim 時間で待っていた。
        time.sleep に置き換えると壁時計で待つことになり、実時間係数が 1 を
        下回る環境 (この構成では 0.26 程度) では待ち時間の意味が変わる。
        そこでノードのクロックで測る。

        sim が一時停止すると sim 時間が進まず永久に待つことになるので、
        壁時計の上限を設けて抜ける。既定では要求の 20 倍か 5 秒の大きい方。
        """
        clock = self.node.get_clock()
        end = clock.now() + Duration(seconds=seconds)
        if wall_timeout is None:
            wall_timeout = max(seconds * 20.0, 5.0)
        wall_start = time.monotonic()
        while clock.now() < end:
            if time.monotonic() - wall_start > wall_timeout:
                self.node.get_logger().warn(
                    f"_sleep({seconds}) が sim 時間で完了しなかった "
                    f"(壁時計 {wall_timeout:.1f} 秒で打ち切り)。"
                    f"Gazebo が一時停止しているか /clock が来ていない。")
                return
            time.sleep(0.001)

    def _ok(self):
        """rospy.is_shutdown() の反転。"""
        return rclpy.ok()

    # ---- 後片付け -------------------------------------------------------

    def close(self):
        """gymnasium.Env.close()。executor スレッドを止めてノードを破棄する。

        これを呼ばないと daemon スレッドがプロセス終了まで残る。
        """
        if getattr(self, "_closed", False):
            return
        self._closed = True

        # 順序が重要。executor を止める前にノードを破棄すると、spin 中の
        # executor が破棄済みのオブジェクトに触って次の通知が出る。
        #   The following exception was never retrieved:
        #   cannot use Destroyable because destruction was requested
        try:
            self._executor.shutdown(timeout_sec=2.0)
        except Exception:
            pass
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        try:
            self._executor.remove_node(self.node)
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass
        if self._owns_context and rclpy.ok():
            rclpy.shutdown()

    
if __name__ == "__main__":
    # rclpy.init() と Node の生成は GazeboEnv.__init__ が行う。
    # rospy.init_node に相当する明示的な初期化は不要。
    env = GazeboEnv()
    try:
        obs, info = env.reset()
        print("reset 完了:", obs.shape, info)
        while env._ok():
            reward, terminated, truncated = env._compute_reward()
            print("Reward:", reward,
                  "terminated:", terminated, "truncated:", truncated)
            env._sleep(0.4)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()



