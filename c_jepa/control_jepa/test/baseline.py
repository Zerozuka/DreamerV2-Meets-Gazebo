import numpy as np
import gymnasium as gym
import torch
from dreamerv2.models.actor import DiscreteActionModel
from dreamerv2.models.rssm import RSSM
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.pixel import ObsDecoder, ObsEncoder
import matplotlib.pyplot as plt
import csv
from gazebo_env import GazeboEnv
from gazebo_wrappers import ImageEnv, OneHotAction
from sensor_msgs.msg import Image
from rosgraph_msgs.msg import Clock

from dreamerv2.training.config_ import RacingCarConfig
from tqdm.auto import tqdm
import pickle
import os
# cv_bridge は使わない。apt の cv_bridge_boost.so が NumPy 1 でコンパイル
# されており NumPy 2 (sionna 2.x が要求) では画像変換が KeyError で落ちる。
# 同等の変換を gz_sionna/src/ros_image.py に自前で持たせた (差分は docstring)。
import cv2
# ROS 1 の rospy を rclpy の上に再現する移植用の層に差し替えている。
# 本来はこのスクリプト自身が Node を持つべきだが、学習済み重みが無くて実行
# 検証できないため、意味を保つ層を挟んで呼び出し側を無改修にしている。
# 詳細は gz_sionna/src/ros1_compat.py の docstring を参照。
#
# gz_sionna はまだ Python モジュールを install していない (dreamerv2 / utils の
# 名前衝突を解消するまで packages=[] のため) ので share のパスを通す。
import os as _os
import sys as _sys
from ament_index_python.packages import get_package_share_directory as _share
_sys.path.insert(0, _os.path.join(_share('gz_sionna'), 'src'))
import ros1_compat as rospy
import pandas as pd 
from nav_msgs.msg import Odometry
from threading import Lock
from wutils.models import Encoder, Predictor, PowerPredictor 
from std_msgs.msg import Float32MultiArray,MultiArrayDimension,Int32
import time
from models.vit_encoder import ViTEncoder
from models.vit_decoder import ViTDecoder
from models.temporal_transformer import TemporalTransformer
from utils.patch_utils import patches_to_image
# paths.py は gz_sionna/src に置いて全パッケージで共有している。
# gz_sionna はまだ Python モジュールを install していないので share のパスを通す。
import os as _p_os
import sys as _p_sys
from ament_index_python.packages import get_package_share_directory as _p_share
_p_sys.path.insert(0, _p_os.path.join(_p_share('gz_sionna'), 'src'))
from ros_image import imgmsg_to_bgr8  # noqa: E402
from paths import model_path, output_path



AE_CKPT = model_path("vit_model", "autoencoder1.pt")
TEMP_CKPT = model_path("vit_model", "temporal1.pt")
# -------------------------
# GLOBALS
# -------------------------
csv_path = output_path("predicted_power_log.csv")
case_id = "case_0/"
output_dir = output_path("baseline") + case_id
_video_writers = {} 
last_completed = None
current_image = None
channels = None

step_2 = 0

prev_frame_global1 = None
prev_frame_global2 = None
prev_frame_global3 = None
prev_frame_global4 = None
prev_frame_global5 = None

# Robot bounding box (your fixed values)
ROBOT_X1, ROBOT_Y1 = 303, 320
ROBOT_X2, ROBOT_Y2 = 341, 412

robot_center_x = (ROBOT_X1 + ROBOT_X2) // 2

# If CSV exists, remove it
if os.path.exists(csv_path):
    os.remove(csv_path)


def clock_callback(msg):
    global sim_time_sec
    sim_time_sec = msg.clock.secs + msg.clock.nsecs * 1e-9


# -------------------------
# CHANNEL CALLBACK
# -------------------------
# def channel_callback(msg):
#     global channels
#     flat = np.array(msg.data, dtype=np.float32)
#     dims = msg.layout.dim

#     channels = dims[0].size  # should be 2
#     d1 = dims[1].size
#     d2 = dims[2].size
#     d3 = dims[3].size

#     stacked = flat.reshape((channels, d1, d2, d3))
#     real = stacked[0]
#     imag = stacked[1]

#     real_torch = torch.from_numpy(real)
#     imag_torch = torch.from_numpy(imag)
#     channels = torch.complex(real_torch, imag_torch)

def channel_callback(msg):
    global channels,step_2
    step_2 += 1
    # Extract raw data
    flat = np.array(msg.data, dtype=np.float32)

    # Read the layout dims
    dims = msg.layout.dim

    num_ch = dims[0].size   # 2 → do NOT overwrite channels
    d1 = dims[1].size       # 3
    d2 = dims[2].size       # 8
    d3 = dims[3].size       # 16

    # Reconstruct stacked tensor (2, 3, 8, 16)
    stacked = flat.reshape((num_ch, d1, d2, d3))

    real = stacked[0]       # (3,8,16)
    imag = stacked[1]       # (3,8,16)

    # Convert numpy to torch
    real_t = torch.from_numpy(real)
    imag_t = torch.from_numpy(imag)

    # Correct: channels becomes complex tensor (3,8,16)
    channels = torch.complex(real_t, imag_t)

# -------------------------
# ODOM LISTENER
# -------------------------
class OdomPoseListener:
    def __init__(self, topic="/odom"):
        self.x = 0.0
        self.y = 0.0
        self.lock = Lock()
        rospy.Subscriber(topic, Odometry, self._callback)

    def _callback(self, msg):
        with self.lock:
            self.x = msg.pose.pose.position.x
            self.y = msg.pose.pose.position.y

    def get_pose(self):
        with self.lock:
            return self.x, self.y



# -------------------------
# IMAGE CALLBACK
# -------------------------
def _image_callback(msg):
    global current_image
    current_image = imgmsg_to_bgr8(msg)



# -------------------------
# PID CONTROLLER
# -------------------------
class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.last_error = 0

    def update(self, error, dt=0.05):
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        self.last_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative



# -------------------------
# LANE DETECTION
# -------------------------
def get_lane_error(img,h_min,h_max,s_min,s_max,v_min,v_max):
    
    h, w, _ = img.shape

    # ROI above robot
    LOOKAHEAD_Y = ROBOT_Y1 - 80
    ROI_HEIGHT = 70

    roi_top = max(0, LOOKAHEAD_Y)
    roi_bottom = min(h, roi_top + ROI_HEIGHT)

    # ------------ NARROW ROI (only center region) ------------
    center = robot_center_x
    half_width = 120  # adjust 60–120 depending on lane width

    left = max(0, center - half_width)
    right = min(w, center + half_width)

    roi = img[roi_top:roi_bottom, left:right]

    # ------------ Detect lane color ------------
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)


    # lower_white = np.array([h_min, s_min, v_min])
    # upper_white = np.array([h_max, s_max, v_max])

    lower_white = np.array([0, 0, 129])
    upper_white = np.array([180, 30,255])
    # lower_white = np.array([0, 0, 200])
    # upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    mask = cv2.GaussianBlur(mask, (5,5), 0)

    M = cv2.moments(mask)
    
    if M["m00"] == 0:
        return None, roi, mask, roi_top

    lane_center_roi = int(M["m10"] / M["m00"])
    lane_center_x = left + lane_center_roi

    error = lane_center_x - robot_center_x
    return error, roi, mask, roi_top



# -------------------------
# PID → CONTINUOUS ACTION (FINAL BASELINE)
# -------------------------
def pid_to_continuous(pid_output, lane_detected):
    BASE_SPEED = 0.9 # fixed forward speed

    if not lane_detected:
        linear = 0.1      # crawl when lane lost
        angular = 0.0
    else:
        linear = BASE_SPEED
        angular = pid_output * 1 # steering scale

    return np.array([linear, angular], dtype=np.float32)

def render_done_cb(msg):
    global last_completed
    last_completed = msg.data


def save_video_frame(img, path, fps=20):
    """Save a single frame to a video file."""
    global _video_writers
    
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if path not in _video_writers:
        h, w = img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        _video_writers[path] = cv2.VideoWriter(path, fourcc, fps, (w, h))
    
    _video_writers[path].write(img)


def image_callback1(msg):
    global prev_frame_global1
    path_ = output_dir + "Video/cam1.mp4" 
    img = imgmsg_to_bgr8(msg)
    # img = cv2.resize(img, (640, 480))

    if prev_frame_global1 is None:
        prev_frame_global1 = img.copy()
        save_video_frame(img, path_)
        return

    # Skip EXACT same frame (Gazebo freeze detection)
    if np.array_equal(img, prev_frame_global1):
        return
    
    save_video_frame(img, path_)
    prev_frame_global1 = img.copy()
 

def image_callback2(msg):
    global prev_frame_global2
    path_ = output_dir + "Video/cam2.mp4" 
    img = imgmsg_to_bgr8(msg)
    # img = cv2.resize(img, (640, 480))

    if prev_frame_global2 is None:
        prev_frame_global2 = img.copy()
        save_video_frame(img, path_)
        return

    # Skip EXACT same frame (Gazebo freeze detection)
    if np.array_equal(img, prev_frame_global2):
        return
    
    save_video_frame(img, path_)
    prev_frame_global2 = img.copy()

def image_callback3(msg):
    global prev_frame_global3
    path_ = output_dir + "Video/cam3.mp4" 
    img = imgmsg_to_bgr8(msg)
    # img = cv2.resize(img, (640, 480))

    if prev_frame_global3 is None:
        prev_frame_global3 = img.copy()
        save_video_frame(img, path_)
        return

    # Skip EXACT same frame (Gazebo freeze detection)
    if np.array_equal(img, prev_frame_global3):
        return
    
    save_video_frame(img, path_)
    prev_frame_global3 = img.copy()
    
def image_callback4(msg):
    global prev_frame_global4
    path_ = output_dir + "Video/cam4.mp4" 
    img = imgmsg_to_bgr8(msg)
    # img = cv2.resize(img, (640, 480))

    if prev_frame_global4 is None:
        prev_frame_global4 = img.copy()
        save_video_frame(img, path_)
        return

    if np.array_equal(img, prev_frame_global4):
        return
    
    save_video_frame(img, path_)
    prev_frame_global4 = img.copy()

def image_callback5(msg):
    global prev_frame_global5
    path_ = output_dir + "Video/cam5.mp4" 
    img = imgmsg_to_bgr8(msg)
    # img = cv2.resize(img, (640, 480))

    if prev_frame_global5 is None:
        prev_frame_global5 = img.copy()
        save_video_frame(img, path_)
        return

    # Skip EXACT same frame (Gazebo freeze detection)
    if np.array_equal(img, prev_frame_global5):
        return
    
    save_video_frame(img, path_)
    prev_frame_global5 = img.copy()


def np_img_to_cv2(img):
    """
    img: numpy HWC float [0,1] or uint8
    """
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    return img[:, :, ::-1]  # RGB → BGR

def torch_img_to_cv2(img):
    """
    img: torch CHW float [0,1]
    """
    img = img.permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img[:, :, ::-1]  # RGB → BGR

def chw_to_hwc(img_chw):
    # img_chw: (3,84,84) in [0,1]
    return img_chw.permute(1, 2, 0).detach().cpu().clamp(0, 1)



def add_random_grass_patches(img, patch_size=40, patch_count=2):
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



def draw_gym_car_shape(image, center=(320, 410), scale=0.25):
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

def transform_image_cv2_2(img, min_road_width=13, close_kernel=9, open_kernel=5, add_grass=False):
   

    if img is None:
        raise FileNotFoundError("Empty frame in transform_image_cv2")

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # --- Step 1: Detect dark/gray road ---
    # lower_road = np.array([0, 0, 20])
    # upper_road = np.array([180, 161, 183])
    lower_road = np.array([0, 0, 44])
    upper_road = np.array([180, 57, 220])
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
        add_random_grass_patches(new_img, patch_size=150, patch_count=3)

    # Paint road gray
    new_img[mask_road > 0] = (102, 102, 102)

    # --- Step 6: Add car representation ---
    draw_gym_car_shape(new_img, center=(320, 410), scale=0.25)

    return new_img

def patched_transform(img):
    
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
    cv2.imshow("resized",img)
    img = transform_image_cv2_2(img)   # expects BGR HWC

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (84, 84))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = img.transpose(1, 2, 0)
    img = cv2.resize(img, (84, 84))
    cv2.imshow("gym_img",img)
    cv2.waitKey(1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0

    return img

def nothing(x):
    pass
# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    
    device = 'cpu'
    rospy.init_node('Baseline', anonymous=True)

    # Ensure Qt GUI thread is started (needed in some ROS/python setups so widgets render)
    cv2.startWindowThread()

    # Use a visible background; some backends draw trackbars only after the first imshow.
    img_src = np.full((320, 700, 3), 200, np.uint8)
    cv2.namedWindow("Trackbars_", cv2.WINDOW_NORMAL)

    cv2.createTrackbar("H min", "Trackbars_", 0,   180, nothing)
    cv2.createTrackbar("H max", "Trackbars_", 0, 180, nothing)
    cv2.createTrackbar("S min", "Trackbars_", 0, 255, nothing)
    cv2.createTrackbar("S max", "Trackbars_", 0, 255, nothing)
    cv2.createTrackbar("V min", "Trackbars_", 0,   255, nothing)
    cv2.createTrackbar("V max", "Trackbars_", 0,  255, nothing)

    # Show the window once so Qt draws the trackbars (prevents them from staying invisible)
    cv2.imshow("Trackbars_", img_src)
    cv2.waitKey(100)

    pose_listener = OdomPoseListener("/odom")
    print("Subscribed to /odom")

    rospy.Subscriber("/channels", Float32MultiArray, channel_callback, queue_size=10)
    rospy.Subscriber("/image_raw2", Image, _image_callback)
    rospy.Subscriber("/clock", Clock, clock_callback)
    rospy.Subscriber("/render_done", Int32, render_done_cb)

    # rospy.Subscriber("/cam_front/world_cam/image_raw", Image, image_callback1)
    # rospy.Subscriber("/cam_back/world_cam/image_raw", Image, image_callback2)
    # rospy.Subscriber("/cam_left/world_cam/image_raw", Image, image_callback3)
    # rospy.Subscriber("/cam_right/world_cam/image_raw", Image, image_callback4)
    # rospy.Subscriber("/cam_top/world_cam/image_raw", Image, image_callback5)
    pred_video0 = output_dir + "Video/pred_frame0.mp4"
    pred_video1 = output_dir + "Video/pred_frame1.mp4"
    flag_pub = rospy.Publisher('/render_trigger', Int32, queue_size=10)
    
    env = GazeboEnv()
    
    datapath = output_dir + "Baseline_results.pt"
    obs, _ = env.reset()

    encoder = ViTEncoder().to(device)
    decoder = ViTDecoder().to(device)
    temporal = TemporalTransformer().to(device)
    

    ae = torch.load(AE_CKPT, map_location=device)
    encoder.load_state_dict(ae["encoder"])
    decoder.load_state_dict(ae["decoder"])


    temporal.load_state_dict(torch.load(TEMP_CKPT, map_location=device))

    encoder.eval()
    decoder.eval()
    temporal.eval()

    while channels is None:
        print("Connecting with Sionna...")

    print("Running PID Baseline...")

    done = False
    pid = PID(kp=0.02, ki=0.0, kd=0.002)

    eval_scores = []
    score = 0
    frame_id = 0
    real_start_time = time.time()
    sim_start_time = None
    prev_act = 0
    dataset = {"step": [],"poses": [], "reward": [],"channels": [],"uplink_com_status": [],"action":[]}
    noise_ratio = 0.000115
    num_power_loss= 0
    rew =0
    terminated = False
    truncated = False
    ROLLOUT_STEPS = 5
    img_h = []
    missing_frame = 0
    kk = 0
    index = 0
    while not rospy.is_shutdown() and not done:
        cv2.imshow('Trackbars_',img_src)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
            # Get trackbar values
        # cv2.waitKey(1)
        h_min = cv2.getTrackbarPos("H min", "Trackbars_")
        h_max = cv2.getTrackbarPos("H max", "Trackbars_")
        s_min = cv2.getTrackbarPos("S min", "Trackbars_")
        s_max = cv2.getTrackbarPos("S max", "Trackbars_")
        v_min = cv2.getTrackbarPos("V min", "Trackbars_")
        v_max = cv2.getTrackbarPos("V max", "Trackbars_")
        

        # dummy = np.zeros((50,300,3), dtype=np.uint8)
        # cv2.imshow("Trackbars_", dummy)
        # cv2.waitKey(1)
        
        if(last_completed is None or frame_id==0):
            channels_t = torch.fft.fft(channels)
            g = channels_t[0]     # shape: [8, 16]

            if current_image is not None:

                if sim_start_time is None and sim_time_sec > 0:
                    sim_start_time = sim_time_sec

                flag_pub.publish(1)
                while (last_completed!=1):
                    # print("loop")
                    af = 0


                g_best = torch.abs(g).max()


                px, py = pose_listener.get_pose()

                orig_show = cv2.resize(current_image,(84,84))
                orig_np = np.transpose(orig_show, (2, 0, 1))      # HWC → CHW
                orig_np = orig_np.astype(np.float32) / 255.0      # normalize
                orig_t = torch.from_numpy(orig_np).unsqueeze(0)   # add batch
                orig_t = orig_t.to(device)

                tokens_ = encoder(orig_t)
                patches_ = decoder(tokens_)            # (1,196,108)
                pred_img_ = patches_to_image(patches_) # (1,3,84,84)
                
                img_ = chw_to_hwc(pred_img_[0].cpu()).numpy()
                img_ = cv2.resize(img_,(640,480))
                img_h = []
                img_h.append(img_)
                

                for step in range(ROLLOUT_STEPS):
                    tokens = temporal(tokens_)            # (1,196,256)

                    patches = decoder(tokens)            # (1,196,108)
                    pred_img = patches_to_image(patches) # (1,3,84,84)
                    img = chw_to_hwc(pred_img[0].cpu()).numpy()
                    img = cv2.resize(img,(640,480))
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_h.append(img)
                
                # ---- SAVE PREDICTED FRAMES ----
                if len(img_h) > 0:
                    frame0 = np_img_to_cv2(img_h[0])
                    save_video_frame(frame0, pred_video0)

                if len(img_h) > 1:
                    frame1 = np_img_to_cv2(img_h[1])
                    save_video_frame(frame1, pred_video1)

                if(frame_id>=130 and frame_id<140):
                    channel_condition = True
                    missing_frame +=1

                else:
                    channel_condition = False


                cv2.imshow("firts",img_h[4])
                #cv2.waitKey(1)

                # error, roi, mask, roi_top = get_lane_error(current_image,h_min,h_max,s_min,s_max,v_min,v_max)
                # cv2.imshow("Lane ROI1", roi)
                # cv2.imshow("Lane Mask1", mask)
                #cv2.waitKey(1)

                error, roi, mask, roi_top = get_lane_error(img_h[0],h_min,h_max,s_min,s_max,v_min,v_max)
                cv2.imshow("Lane ROI", roi)
                cv2.imshow("Lane Mask", mask)
                #cv2.waitKey(1)
                if error is None:
                    lane_detected = False
                    pid_output = 0
                else:
                    lane_detected = True
                    pid_output = pid.update(error)

                # Continuous action (BASELINE OUTPUT):
                action = pid_to_continuous(pid_output, lane_detected)

                next_obs, rew, terminated, truncated, info = env.step(action)
                rew +=rew

                print(f"[Frame {frame_id}] Action init = {action}")
                # Step environment
                # for i in range(3):
                if kk %3==0:
                    


                    dataset["step"].append(frame_id)
                    dataset["poses"].append([float(px), float(py)])
                    dataset["reward"].append(rew)
                    dataset["channels"].append(channels_t)
                    dataset["action"].append(action)
                    frame_id += 1
                    score += rew
                    rew = 0
                    
                if terminated or truncated:
                    print("Episode finished.")
                    done = True
                else:
                    done = False


                

                kk +=1

                last_completed = None
                prev_act = action
                
    cv2.destroyAllWindows()

    eval_scores.append(score)
    avg_score = np.mean(eval_scores)

    real_total_time = time.time() - real_start_time
    sim_total_time = sim_time_sec - sim_start_time if sim_start_time else 0

    timing_path = output_dir+"timing_info.txt"

    os.makedirs(os.path.dirname(timing_path), exist_ok=True)

    with open(timing_path, "w") as f:
        f.write(f"Real time (seconds): {real_total_time}\n")
        f.write(f"Simulation time (seconds): {sim_total_time}\n")
        f.write(f"Total Reward: {avg_score}\n")
        f.write(f"Total frames: {frame_id}\n")
        f.write(f"Missing frames: {missing_frame}\n")

    print("Saved timing info to:", timing_path)

    torch.save(dataset, datapath)
    print("Saved dataset with", len(dataset["poses"]), "samples")
    print("Total frames:", frame_id)
    print(f"Average evaluation score = {avg_score}")
    print("Total steps loss power", num_power_loss)
