#!/usr/bin/env python3
# coding: utf-8

import gymnasium as gym
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from gazebo_env import GazeboEnv
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
from nav_msgs.msg import Odometry
from threading import Lock
from wutils.models import Encoder, Predictor, PowerPredictor 
from std_msgs.msg import Float32MultiArray,MultiArrayDimension,Int32
import time
import os
# cv_bridge は使わない。apt の cv_bridge_boost.so が NumPy 1 でコンパイル
# されており NumPy 2 (sionna 2.x が要求) では画像変換が KeyError で落ちる。
# 同等の変換を gz_sionna/src/ros_image.py に自前で持たせた (差分は docstring)。
from sensor_msgs.msg import Image
from rosgraph_msgs.msg import Clock
# paths.py は gz_sionna/src に置いて全パッケージで共有している。
# gz_sionna はまだ Python モジュールを install していないので share のパスを通す。
import os as _p_os
import sys as _p_sys
from ament_index_python.packages import get_package_share_directory as _p_share
_p_sys.path.insert(0, _p_os.path.join(_p_share('gz_sionna'), 'src'))
from ros_image import imgmsg_to_bgr8  # noqa: E402
from paths import output_path



csv_path = output_path("predicted_power_log.csv")
case_id = "case_8/"
output_dir = output_path("baseline_dqn") + case_id
_video_writers = {} 
last_completed = None
current_image = None
channels = None

prev_frame_global1 = None
prev_frame_global2 = None
prev_frame_global3 = None
prev_frame_global4 = None
prev_frame_global5 = None


# ============================================================
# Preprocessing (SAME AS TRAINING)
# ============================================================

def preprocess(img):
    """
    Handles:
    - RGB HWC  (H, W, 3)
    - RGB CHW  (3, H, W)
    - Grayscale (H, W)
    """
    # If CHW -> HWC
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC

    # Resize (safe for both)
    img = cv2.resize(img, (84, 84))
    cv2.imshow("img",img)
    cv2.waitKey(1)
    # If RGB -> grayscale
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = img.astype(np.float32) / 255.0
    return img

# ============================================================
# Image Wrapper (SAME AS TRAINING)
# ============================================================

class ImageEnv(gym.Wrapper):
    def __init__(self, env, skip_frames=3 ,stack_frames=4, initial_no_op=5):
        super().__init__(env)
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
        self.initial_no_op = initial_no_op

    def reset(self):
        s, info = self.env.reset()
        for _ in range(self.initial_no_op):
            s, _, terminated, truncated, _ = self.env.step(0)
            if terminated or truncated:
                s, info = self.env.reset()
        s = preprocess(s)
        self.stacked_state = np.tile(s, (self.stack_frames, 1, 1))
        return self.stacked_state, info

    def step(self, action):
        reward = 0.0
        for _ in range(self.skip_frames):
            s, r, terminated, truncated, info = self.env.step(action)
            reward += r
            if terminated or truncated:
                break
        s = preprocess(s)
        self.stacked_state = np.concatenate(
            (self.stacked_state[1:], s[np.newaxis]), axis=0
        )
        return self.stacked_state, reward, terminated, truncated, info

# ============================================================
# Network (SAME AS TRAINING)
# ============================================================

class CNNActionValue(nn.Module):
    def __init__(self, in_channels, action_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 8, 4)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def clock_callback(msg):
    global sim_time_sec
    sim_time_sec = msg.clock.secs + msg.clock.nsecs * 1e-9


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



def channel_callback(msg):
    global channels

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

def get_base_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env

if __name__ == "__main__":
    rospy.init_node('Gazebo_test', anonymous=True)
    DEVICE = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    flag_pub = rospy.Publisher('/render_trigger', Int32, queue_size=10)


    MODEL_PATH = "training/gazebo/gazebo_step_230000.pt"    #gazebo_step_50000 gazebo_best

    env = ImageEnv(GazeboEnv())

    state_dim = (4, 84, 84)
    action_dim = env.action_space.n
    print(action_dim)
    model = CNNActionValue(state_dim[0], action_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("Model loaded successfully "+MODEL_PATH)

    while channels is None:
        print("Connecting with Sionna...")

    datapath = output_dir + "Baseline_results.pt"

    state, _ = env.reset()
    done = False
    total_reward = 0.0

    eval_scores = []
    score = 0
    frame_id = 0
    real_start_time = time.time()
    sim_start_time = None
    prev_act = 0
    dataset = {"step": [],"poses": [], "reward": [],"channels": [],"uplink_com_status": [],"action":[]}
    noise_ratio = 0.000115
    num_power_loss= 0

    base_env = get_base_env(env)
    base_env.set_visual_mode(0)
    while not rospy.is_shutdown() and not done:

        if(last_completed is None or frame_id==0):
            channels_t = torch.fft.fft(channels)
            g = channels_t[0]     # shape: [8, 16]
            source_img,timg = base_env.get_obs_with_patch(0,False)
            if current_image is not None:

                if sim_start_time is None and sim_time_sec > 0:
                    sim_start_time = sim_time_sec

                with torch.no_grad():

                    flag_pub.publish(1)
                    while (last_completed!=1):
                        af = 0


                    g_best = torch.abs(g).max()

                    channel_condition = (noise_ratio > g_best.item())
                    px, py = pose_listener.get_pose()
        
                    s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                    action = model(s).argmax().item()
                    channel_condition = False
                    if(channel_condition==True):
                        action = prev_act
                        num_power_loss += 1
                        print(channel_condition, g_best)
                        dataset["uplink_com_status"].append(0)
                    else:
                        dataset["uplink_com_status"].append(1)

                    # print(model(s))
                    # print(action)

                    state, reward, terminated, truncated, _ = env.step(action)

                    dataset["step"].append(frame_id)
                    dataset["poses"].append([float(px), float(py)])
                    dataset["reward"].append(reward)
                    dataset["channels"].append(channels_t)
                    dataset["action"].append(action)

                    if terminated or truncated:
                        print("Episode finished.")
                        done = True
                    else:
                        done = False

                    frame_id += 1
                    score += reward

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

    print("Saved timing info to:", timing_path)

    torch.save(dataset, datapath)
    print("Saved dataset with", len(dataset["poses"]), "samples")
    print("Total frames:", frame_id)
    print(f"Average evaluation score = {avg_score}")
    print("Total steps loss power", num_power_loss)
