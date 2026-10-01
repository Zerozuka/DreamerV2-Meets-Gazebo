import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from dreamerv2.models.actor import DiscreteActionModel
from dreamerv2.models.rssm import RSSM
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.pixel import ObsDecoder, ObsEncoder
import matplotlib.pyplot as plt
import csv
from gazebo_env import GazeboEnv
from gazebo_wrappers import ImageEnv, OneHotAction
# cv_bridge は使わない。apt の cv_bridge_boost.so が NumPy 1 でコンパイル
# されており NumPy 2 (sionna 2.x が要求) では画像変換が KeyError で落ちる。
# 同等の変換を gz_sionna/src/ros_image.py に自前で持たせた (差分は docstring)。
import time
from dreamerv2.training.config_ import RacingCarConfig
from tqdm.auto import tqdm
import pickle
import os
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
from sensor_msgs.msg import Image
from rosgraph_msgs.msg import Clock
from numpy.linalg import norm
import torch.nn.functional as F
import random
import torch.distributions as dist
# paths.py は gz_sionna/src に置いて全パッケージで共有している。
# gz_sionna はまだ Python モジュールを install していないので share のパスを通す。
import os as _p_os
import sys as _p_sys
from ament_index_python.packages import get_package_share_directory as _p_share
_p_sys.path.insert(0, _p_os.path.join(_p_share('gz_sionna'), 'src'))
from ros_image import imgmsg_to_bgr8  # noqa: E402
from paths import model_path, output_path


MUD_COLOR = (14, 34, 49) #(50,50,47) #(14, 34, 49)  # BGR (OpenCV)
episode_id = 0
seed = 42

csv_path = output_path("predicted_power_log.csv")
case_id = "case_0/"
output_dir = output_path("Proposed") + case_id
datapath = output_dir + "proposed_results.pt"
datapath_2 = output_dir + "z_val_.pt"
_video_writers = {} 
last_completed = None

prev_frame_global1 = None
prev_frame_global2 = None
prev_frame_global3 = None
prev_frame_global4 = None
prev_frame_global5 = None

mean_val = 0.009106356651
var_val = 0.2119901876

if os.path.exists(csv_path):
    os.remove(csv_path)

channels = None


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 84, 84)
            h = self.encoder(dummy)
            self.enc_shape = h.shape[1:]
            self.flat_dim = h.numel()

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        h = self.fc_dec(z)
        h = h.view(-1, *self.enc_shape)

        recon = self.decoder(h)
        recon = F.interpolate(
            recon, size=(84, 84),
            mode="bilinear",
            align_corners=False
        )
        return recon
    


# class Encoder(nn.Module):
#     def __init__(self, latent_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),   # 84 → 42
#             nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(), # 42 → 21
#             nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),# 21 → 10
#             nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),# 10 → 5
#             nn.Flatten()
#         )
#         self.fc = nn.Linear(256 * 5 * 5, 2 * latent_dim)

#     def forward(self, x):
#         return self.fc(self.net(x))


# class Decoder(nn.Module):
#     def __init__(self, latent_dim):
#         super().__init__()
#         self.fc = nn.Linear(latent_dim, 256 * 5 * 5)
#         self.net = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
#             nn.ConvTranspose2d(32, 3, 4, 2, 1),
#         )

#     def forward(self, z):
#         h = self.fc(z).view(-1, 256, 5, 5)
#         x_mu = self.net(h)
#         return F.interpolate(x_mu, size=(84, 84), mode="bilinear", align_corners=False)


# class VAE(nn.Module):
#     def __init__(self, latent_dim):
#         super().__init__()
#         self.encoder = Encoder(latent_dim)
#         self.decoder = Decoder(latent_dim)

#     def forward(self, x):
#         mu_logvar = self.encoder(x)
#         mu, logvar = torch.chunk(mu_logvar, 2, dim=1)
#         std = torch.exp(0.5 * logvar)

#         qz = dist.Normal(mu, std)
#         z = qz.rsample()

#         x_mu = self.decoder(z)
#         return x_mu, mu, logvar



def render_done_cb(msg):
    global last_completed
    last_completed = msg.data


def save_video_frame(img, path, fps=15):
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

def clock_callback(msg):
    global sim_time_sec
    sim_time_sec = msg.clock.secs + msg.clock.nsecs * 1e-9

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



# def channel_callback(msg):
#     global channels
#     # Extract raw data
#     flat = np.array(msg.data, dtype=np.float32)

#     # Read the layout dims
#     dims = msg.layout.dim

#     # Expected layout:
#     # dim[0] -> channels = 2 (real, imag)
#     # dim[1] -> d1 = 3
#     # dim[2] -> d2 = 8
#     # dim[3] -> d3 = 16

#     channels = dims[0].size  # should be 2
#     d1 = dims[1].size         # 3
#     d2 = dims[2].size         # 8
#     d3 = dims[3].size         # 16

#     # Reconstruct stacked tensor (2,3,8,16)
#     stacked = flat.reshape((channels, d1, d2, d3))

#     real = stacked[0]        # (3,8,16)
#     imag = stacked[1]        # (3,8,16)

#     # Build complex64 array
#     real_torch = torch.from_numpy(real)      # convert numpy → torch
#     imag_torch = torch.from_numpy(imag)
#     channels = torch.complex(real_torch, imag_torch) 
    


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
        


def load_model(config, model_path, device):
    saved_dict = torch.load(model_path,map_location=torch.device("cpu"),weights_only=True)
    obs_shape = config.obs_shape
    action_size = config.action_size
    deter_size = config.rssm_info['deter_size']

    if config.rssm_type == 'continuous':
        stoch_size = config.rssm_info['stoch_size']
    elif config.rssm_type == 'discrete':
        category_size = config.rssm_info['category_size']
        class_size = config.rssm_info['class_size']
        stoch_size = category_size * class_size
    else:
        raise ValueError(f"Unknown RSSM type: {config.rssm_type}")

    embedding_size = config.embedding_size
    rssm_node_size = config.rssm_node_size
    modelstate_size = stoch_size + deter_size

    # Encoder/Decoder setup
    if config.pixel:
        ObsEncoderModel = ObsEncoder(obs_shape, embedding_size, config.obs_encoder).to(device).eval()
        ObsDecoderModel = ObsDecoder(obs_shape, modelstate_size, config.obs_decoder).to(device).eval()
    else:
        ObsEncoderModel = DenseModel((embedding_size,), int(np.prod(obs_shape)), config.obs_encoder).to(device).eval()
        ObsDecoderModel = DenseModel(obs_shape, modelstate_size, config.obs_decoder).to(device).eval()

    # Actor and RSSM setup
    ActionModel = DiscreteActionModel(
        action_size, deter_size, stoch_size, embedding_size, config.actor, config.expl
    ).to(device).eval()
    RSSMModel = RSSM(
        action_size, rssm_node_size, embedding_size, device, config.rssm_type, config.rssm_info
    ).to(device).eval()

    # Load model states
    RSSMModel.load_state_dict(saved_dict["RSSM"])
    ObsEncoderModel.load_state_dict(saved_dict["ObsEncoder"])
    ActionModel.load_state_dict(saved_dict["ActionModel"])

    return RSSMModel, ObsEncoderModel, ActionModel

def load_wireless_model(model_path, device):
    saved_dict = torch.load(model_path,map_location=torch.device("cpu"),weights_only=True)
    encoder = Encoder().to(device)
    predictor = Predictor(input_dim=1324, hidden_dim=256, num_layers=1, output_dim=2).to(device)
    power_predictor = PowerPredictor().to(device)

    encoder.load_state_dict(saved_dict["encoder"])
    predictor.load_state_dict(saved_dict["predictor"])
    power_predictor.load_state_dict(saved_dict["power"])

    return encoder , predictor , power_predictor


def print_shape(name, x):
    print(f"\n{name}:")
    print("  type :", type(x))

    if isinstance(x, list):
        print("  length :", len(x))
        if len(x) > 0:
            print("  element[0] type :", type(x[0]))
            if hasattr(x[0], "shape"):
                print("  element[0] shape:", x[0].shape)
    else:
        if hasattr(x, "shape"):
            print("  shape:", x.shape)

def to_vae_input(img, device):
    """
    img: numpy HWC uint8/float OR torch CHW/BCHW
    returns: torch BCHW float32 in [0,1]
    """
    if isinstance(img, np.ndarray):
        t = torch.from_numpy(img).float()
        # HWC -> CHW
        if t.ndim == 3:
            t = t.permute(2, 0, 1)
        # normalize if 0..255
        if t.max() > 1.0:
            t = t / 255.0
        # add batch
        t = t.unsqueeze(0)
    else:
        t = img
        if t.ndim == 3:
            t = t.unsqueeze(0)
        t = t.float()
        if t.max() > 1.0:
            t = t / 255.0

    return t.to(device)

def vae_out_to_cv2(recon):
    """
    recon: torch BCHW in [0,1]
    returns: numpy HWC uint8 BGR (for cv2.imshow)
    """
    recon_np = recon[0].permute(1, 2, 0).detach().cpu().numpy()  # HWC float
    recon_np = (recon_np * 255).clip(0, 255).astype(np.uint8)
    return recon_np  # still BGR ordering like your training code



def np_img_to_cv2(img):
    """
    img: numpy HWC float [0,1] or uint8
    """
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    return img  #[:, :, ::-1]  # RGB → BGR

def torch_img_to_cv2(img):
    """
    img: torch CHW float [0,1]
    """
    img = img.permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img  #[:, :, ::-1]

def add_mud_patch(image):
    """
    image: torch.Tensor [3,84,84] or numpy [84,84,3]
    returns: torch.Tensor [3,84,84]
    """

    # ---- to numpy HWC uint8 ----
    rng = random.Random(seed)
    if isinstance(image, torch.Tensor):
        img = (image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    else:
        img = image.copy()
        if img.dtype != np.uint8:
            img = (img * 255).astype("uint8")

    h, w, _ = img.shape
    overlay = img.copy()

    num_patches = rng.randint(7, 10)
    for _ in range(num_patches):
        cx = rng.randint(0, w - 1)
        cy = rng.randint(0, h - 1)
        r  = rng.randint(15, 20)
        cv2.circle(overlay, (cx, cy), r, MUD_COLOR, -1)

    alpha = 1.0  # hard mud (can change to 0.6 if needed)
    muddy = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # ---- back to torch CHW float ----
    muddy = torch.from_numpy(muddy).permute(2, 0, 1).float() / 255.0
    return muddy

def dbg(name, img):
    print(
        name,
        "type:", type(img),
        "shape:", getattr(img, "shape", None),
        "dtype:", getattr(img, "dtype", None),
        "min/max:",
        (img.min(), img.max()) if isinstance(img, np.ndarray) else "N/A"
    )

def visualize_gray(img, win, resize=1):
    """
    img: (H,W) or (1,H,W) numpy float or uint8
    """
    if img.ndim == 3:
        img = img[0]  # remove batch

    img = img.astype(np.float32)

    # normalize for display
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)

    if resize > 1:
        img = cv2.resize(img, (img.shape[1]*resize, img.shape[0]*resize),
                         interpolation=cv2.INTER_NEAREST)

    cv2.imshow(win, img)



def eval_saved_agent(env, config, model_path,wmodel_path ,device,flag_pub,vae_model):
    global channels,last_completed
    RSSMModel, ObsEncoderModel, ActionModel = load_model(config, model_path, device)
    encoder , predictor , power_predictor = load_wireless_model(wmodel_path, device)
    eval_episode = config.eval_episode
    eval_scores = []
    action_size = config.action_size
    dataset = {"step": [],"srcimage": [],"transimage": [],"poses": [],"recon_loss":[],"mud_loss":[],"prev_action":[], "reward": [],"channels": [],"csi_embedding": [],"latent_state": [],"action": [],"predicted_power": [],"uplink_com_status": [],"color_area_id":[],"prev_rssmstate":[],"model_state":[],"model_state1":[],"model_state2":[],"rssm_state":[],"rssm_state1":[],"rssm_state2":[]}
    testdata = {"step": [],"z_value": [],"h_value":[],"rssm_state":[],"z_h":[]}
    # for e in range(eval_episode):

    base_env = get_base_env(env)
    base_env.set_visual_mode(0)

    obs, info = env.reset()
    score = 0
    done = False
    global episode_id
    episode_id += 1
    random.seed(episode_id)

    prev_rssmstate = RSSMModel._init_rssm_state(1)
    prev_action = torch.zeros(1, action_size).to(device)
    print("in the function")
    flag_ = True

    frame_id = 0
    log_data = []  # list of dicts to collect info
    rew = 0
    steps = 0
    # print(channels[:, 0, ...].cfloat())
    encoder.eval()
    horizon_len = 4

    real_start_time = time.time()
    sim_start_time = None
    init_steps = 2000000
    flag_h = True
    sequence_mode = False
    sequence_index = 0
    sequence_target = 0

    main_act_horizon = None
    main_power_h = None
    main_prev_rssmstate = None
    run_jepa = True
    noise_ratio = 0.000115
    prv_main_act_h = []
    prv_sq_indx = 0
    reset_flag = False
    reset_indx = 0
    prv_main_rxxm_h = []
    horizon_flag = False


    
    csv_rows = []

    prev_probs = None
    prev_h = None
    lighting_counter = 0

    KL_THRESH = 0.3        
    H_THRESH  = 0.013333333 #0.02      
    CONFIRM_STEPS = 3

    timg = None
    source_img = None
    color_idx = 0
    mse_zpatched = []
    mse_recon = []

    path_mud = output_dir + "Video/cam_mud.mp4" 
    path_recon = output_dir + "Video/cam_recon.mp4" 
    patch_add_idx = 280
    z_prev = None
    z_prev_h = []
    error_ = []
    prev_obs = None 
    latent_error_ = []
    checked_patched =  False
    while not rospy.is_shutdown() and not done:
        # last_completed = None
        
        if (last_completed is None or frame_id == 0):

            chan_fft = torch.fft.fft(channels)   # was: channels = torch.fft.fft(channels)
            g = chan_fft[0]
            p_flat = chan_fft.reshape(chan_fft.size(0), -1)

            if sim_start_time is None and sim_time_sec > 0:
                sim_start_time = sim_time_sec
            
            with torch.no_grad():
                
                flag_pub.publish(1)
                print_flag = 0
                while (last_completed!=1):
                    if(print_flag == 0):
                        print("loop")
                        print_flag = 1
                    af = 0

                last_completed = None

                if(run_jepa == True):

                
                    source_img,timg = base_env.get_obs_with_patch(0,False)
                    

                    if(frame_id<patch_add_idx):
                        orig_show  = np_img_to_cv2(source_img)
                        original_t = base_env.patched_transform(orig_show)
                        original_t_ = np.expand_dims(original_t,axis=0)
                        obs_new= original_t_
                        save_img = orig_show
                    
                    else:
                        muddy_ = add_mud_patch(source_img)
                        muddy_show = torch_img_to_cv2(muddy_)
                        muddy_t = base_env.patched_transform(muddy_show)
                        muddy_t_ = np.expand_dims(muddy_t,axis=0)
                        obs_new= muddy_t_
                        save_img = muddy_show



                    save_video_frame(save_img, path_mud)
                    


                    embed = ObsEncoderModel(torch.tensor(obs_new, dtype=torch.float32)
                                            .unsqueeze(0).to(device))

                    _, posterior_rssm_state = RSSMModel.rssm_observe(
                        embed, prev_action, not done, prev_rssmstate
                    )
                    model_state = RSSMModel.get_model_state(posterior_rssm_state)
                    action, _ = ActionModel(model_state)


                    z_dim = 1023                
                    z_current = model_state[:, :z_dim] 
                    z_h = []
                    z_h.append(z_current)

                    if(frame_id>10):
                        latent_error = F.mse_loss(z_current, z_prev).item()
                        latent_error2 = F.mse_loss(z_current, z_prev_h[1]).item()
                        error = 1*latent_error+1*latent_error2
                        latent_error_.append(error)

                        obs_torch = torch.from_numpy(obs_new).float().unsqueeze(0)
                        prev_obs_torch = torch.from_numpy(prev_obs).float().unsqueeze(0)

                        
                        obs_error =  F.mse_loss(obs_torch, prev_obs_torch).item()
                        error_.append(obs_error)

                        print(obs_error)
                        if(obs_error>1e-7 or checked_patched==True):
                            print("Mud images")
                            muddy = muddy_.unsqueeze(0).to(device)
                            recon = vae_model(muddy)
                            checked_patched = True
                            recon_show = vae_out_to_cv2(recon)
                            recon_t = base_env.patched_transform(recon_show)
                            recon_t_ = np.expand_dims(recon_t,axis=0)
                            save_video_frame(recon_show, path_recon)
                            obs_new = recon_t_

                            embed = ObsEncoderModel(torch.tensor(obs_new, dtype=torch.float32)
                                            .unsqueeze(0).to(device))

                            _, posterior_rssm_state = RSSMModel.rssm_observe(
                                embed, prev_action, not done, prev_rssmstate
                            )
                            model_state = RSSMModel.get_model_state(posterior_rssm_state)
                            action, _ = ActionModel(model_state)
                            
                            
                    # recon = vae_model(muddy)
                    # recon_show = vae_out_to_cv2(recon)
                    # recon_t = base_env.patched_transform(recon_show)
                    # recon_t_ = np.expand_dims(recon_t,axis=0)
                    # save_video_frame(recon_show, path_recon)

                    # embed2 = ObsEncoderModel(torch.tensor(obs, dtype=torch.float32)
                    #                         .unsqueeze(0).to(device))

                    # _, posterior_rssm_state2 = RSSMModel.rssm_observe(
                    #     embed2, prev_action, not done, prev_rssmstate
                    # )
                    # model_state2 = RSSMModel.get_model_state(posterior_rssm_state2)
                    # action2, _ = ActionModel(model_state2)



                    # embed3 = ObsEncoderModel(torch.tensor(muddy_t_, dtype=torch.float32)
                    #                         .unsqueeze(0).to(device))

                    # _, posterior_rssm_state3 = RSSMModel.rssm_observe(
                    #     embed3, prev_action, not done, prev_rssmstate
                    # )
                    # model_state3 = RSSMModel.get_model_state(posterior_rssm_state3)
                    # action3, _ = ActionModel(model_state3)


                    # recon_t_t = torch.from_numpy(recon_t_).float().to(device)
                    # obs_t     = torch.from_numpy(obs).float().to(device)

                    # # make batch dimensions match
                    # if recon_t_t.ndim == 3:
                    #     recon_t_t = recon_t_t.unsqueeze(0)
                    # if obs_t.ndim == 3:
                    #     obs_t = obs_t.unsqueeze(0)


                    # z_dim = 1023                
                    # z_original = model_state2[:, :z_dim] 
                    # z_recon = model_state[:, :z_dim]
                    # z_patched = model_state3[:, :z_dim]

                    # recon_loss = F.mse_loss(z_original, z_recon)
                    # mud_loss = F.mse_loss(z_original, z_patched)

                    # mse_recon.append(recon_loss.item())
                    # mse_zpatched.append(mud_loss.item())


                    # base_env.set_visual_mode(0)
                    chan_batch = chan_fft.unsqueeze(0)   # was: channels = channels.unsqueeze(0)

                    latent_dynamics = model_state.unsqueeze(0)

                    csi_embed = encoder(chan_batch.cfloat())  # was: encoder(channels.cfloat())
                    predictions, _ = predictor(latent_dynamics.float())
                    predictions = predictions + csi_embed.unsqueeze(dim=1) 

                    predicted_power = power_predictor(predictions)
                    pred_power_value = predicted_power.squeeze().cpu().numpy()

                    action_horizon = []
                    power_pred_horizon = []
                    model_state_horizon = []
                    prev_rssmstate_h = []
                    action_horizon.append(action)
                    power_pred_horizon.append(pred_power_value)
                    model_state_horizon.append(model_state.squeeze(0).cpu().numpy())
                    prev_rssmstate = posterior_rssm_state
                    prev_action = action
                    prev_rssmstate_h.append(prev_rssmstate)

                    for t in range(horizon_len-1):
                        posterior_rssm_state = RSSMModel.rssm_imagine(prev_action, prev_rssmstate)
                        model_state = RSSMModel.get_model_state(posterior_rssm_state)
                        action, _ = ActionModel(model_state)
                        z_dim = 1023                
                        z_current = model_state[:, :z_dim] 
                        z_h.append(z_current)

                        latent_dynamics = model_state.unsqueeze(0)
                        predictions_, _ = predictor(latent_dynamics.float())
                        predictions += predictions_
                        predicted_power = power_predictor(predictions)
                        pred_power_value = predicted_power.squeeze().cpu().numpy()
                        
                        power_pred_horizon.append(pred_power_value)
                        action_horizon.append(action)
                        model_state_horizon.append(model_state.squeeze(0).cpu().numpy())
                    
                        prev_rssmstate = posterior_rssm_state
                        prev_action = action
                        prev_rssmstate_h.append(prev_rssmstate)

                    z_prev = z_current
                    z_prev_h = z_h
                    prev_obs = obs_new

                g_best = torch.abs(g).max()
                channel_condition = (noise_ratio > g_best.item())

                px, py = pose_listener.get_pose()


                if not sequence_mode:
                    # update main horizon/power
                    main_act_horizon = action_horizon
                    main_power_h = power_pred_horizon
                    main_prev_rssmstate = prev_rssmstate_h

                    if frame_id<init_steps:

                        


                        print(f"[Frame {frame_id}] Action init = {action_horizon[0]}")
                        next_obs, rew, terminated, truncated, info = env.step(action_horizon[0].squeeze(0).cpu().numpy())

                        dataset["srcimage"].append(source_img)
                        dataset["transimage"].append(timg)
                        dataset["step"].append(frame_id)
                        dataset["poses"].append([float(px), float(py)])
                        dataset["reward"].append(rew)
                        dataset["channels"].append(chan_fft)
                        dataset["csi_embedding"].append(csi_embed.squeeze(0).cpu().numpy())
                        dataset["latent_state"].append(model_state_horizon)
                        dataset["action"].append(action_horizon)
                        dataset["predicted_power"].append(power_pred_horizon)
                        dataset["uplink_com_status"].append(int(1))
                        dataset["color_area_id"].append(color_idx)
                        dataset["prev_rssmstate"].append(prev_rssmstate)
                        dataset["prev_action"].append(prev_action)
                        dataset["model_state"].append(model_state.squeeze(0).cpu().numpy())
                        # dataset["mud_loss"].append(mud_loss.item())
                        # dataset["recon_loss"].append(recon_loss.item())
                        dataset["rssm_state"].append(posterior_rssm_state)
                        # dataset["rssm_state1"].append(posterior_rssm_state1)
                        # dataset["rssm_state2"].append(posterior_rssm_state2)
                        

                        prev_rssmstate = prev_rssmstate_h[0]
                        prev_action = action_horizon[0]
                        frame_id += 1
                        score += rew
                        obs = next_obs
                        last_completed = None

                        if(truncated==True or terminated==True):
                            print("loop truncated")
                            done = True
                    
                            
                        else:
                            done = False
                    
                    else:
                        
                        min_power = min(main_power_h)
                        min_index = main_power_h.index(min_power)
                        sequence_target = min_index
                        sequence_index = 0
                        sequence_mode = True
                        flag_h = False
                        run_jepa = False
                        horizon_flag = True



                if(flag_h==False):  # Horizon mode
                    
                    temp_act_h = action_horizon
                    temp_power = power_pred_horizon
                    # channel_condition = False
                    if channel_condition == True and horizon_flag==True:
                        print("Power max: ",prv_sq_indx)
                        reset_flag = True
                        dataset["uplink_com_status"].append(int(0))
                        if((prv_sq_indx)==horizon_len):
                            reset_indx = prv_sq_indx-1
                            next_obs, rew, terminated, truncated, info = env.step(prv_main_act_h[prv_sq_indx-1].squeeze(0).cpu().numpy())
                            print(f"[Frame {frame_id}] Action  = {prv_main_act_h[prv_sq_indx-1]}")
                        else:
                            reset_indx=prv_sq_indx
                            next_obs, rew, terminated, truncated, info = env.step(prv_main_act_h[prv_sq_indx].squeeze(0).cpu().numpy())
                            print(f"[Frame {frame_id}] Action  = {prv_main_act_h[prv_sq_indx]}")
                        horizon_flag = False

                    else:
                        next_obs, rew, terminated, truncated, info = env.step(main_act_horizon[sequence_index].squeeze(0).cpu().numpy())
                        if sequence_index ==0:
                            dataset["uplink_com_status"].append(int(1))
                        else:
                            dataset["uplink_com_status"].append(int(0))

                        print(f"[Frame {frame_id}] Action  = {main_act_horizon[sequence_index]}")
                        horizon_flag = False
                    dataset["srcimage"].append(source_img)
                    dataset["transimage"].append(timg)
                    dataset["step"].append(frame_id)
                    dataset["poses"].append([float(px), float(py)])
                    dataset["reward"].append(rew)
                    dataset["channels"].append(chan_fft)
                    dataset["csi_embedding"].append(csi_embed.squeeze(0).cpu().numpy())
                    dataset["latent_state"].append(model_state_horizon)
                    dataset["action"].append(action_horizon)
                    dataset["predicted_power"].append(power_pred_horizon)
                    dataset["color_area_id"].append(color_idx)
                    dataset["prev_rssmstate"].append(prev_rssmstate)

                    
                    sequence_index += 1

                    frame_id += 1
                    score += rew
                    obs = next_obs
                    last_completed = None

                    if (sequence_index > sequence_target) or reset_flag==True:
                        if(reset_flag==True):
                            prev_rssmstate = prv_main_rxxm_h[reset_indx]  #reset_indx
                            prev_action = prv_main_act_h[reset_indx]
                    
                        else:
                            prev_rssmstate = main_prev_rssmstate[sequence_target] #sequence_target
                            prev_action = main_act_horizon[sequence_target]

                        sequence_mode = False
                        flag_h = True
                        run_jepa = True
                        prv_sq_indx = sequence_index
                        prv_main_act_h = main_act_horizon
                        prv_main_rxxm_h = main_prev_rssmstate
                        reset_flag = False

            if(truncated==True or terminated==True):
                print("loop truncated")
                done = True

        
            else:
                done = False
        

    eval_scores.append(score)
    avg_score = np.mean(eval_scores)

    real_total_time = time.time() - real_start_time
    sim_total_time = sim_time_sec - sim_start_time if sim_start_time else 0

    df = pd.DataFrame(csv_rows)
    df.to_csv("stoch_z_stats.csv", index=False)
    timing_path = output_dir+"timing_info.txt"
    os.makedirs(os.path.dirname(timing_path), exist_ok=True)
    with open(timing_path, "w") as f:
        f.write(f"Real time (seconds): {real_total_time}\n")
        f.write(f"Simulation time (seconds): {sim_total_time}\n")
        f.write(f"Total Reward: {avg_score}\n")
        f.write(f"Total frames: {frame_id}\n")
        f.write(f"original_z and recon_z: {np.mean(mse_recon)}\n")
        f.write(f"original_z and mud_z: {np.mean(mse_zpatched)}\n")

    print("Saved timing info to:", timing_path)

    torch.save(dataset, datapath)
    print("Saved dataset with", len(dataset["poses"]), "samples")
    print("frame_id ", frame_id)
    print(f'Average evaluation score for model at = {avg_score}')
    env.close()

    print(np.mean(mse_recon))
    print(np.mean(mse_zpatched))
    plt.figure(figsize=(8,5))
    plt.plot(error_, label="original_z and recon_z")
    plt.plot(mse_zpatched, label="original_z and mud_z")
    plt.xlabel("steps")
    plt.ylabel("Loss")
    plt.title("Latent Z error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(8,5))
    plt.plot(latent_error_, label="original_z and recon_z")
    plt.plot(mse_zpatched, label="original_z and mud_z")
    plt.xlabel("steps")
    plt.ylabel("Loss")
    plt.title("Latent Z error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    return avg_score


def count_parameters(model):
    """Returns total number of trainable parameters and estimated size in MB."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Assuming float32 (4 bytes per parameter)
    size_mb = total_params * 4 / (1024 ** 2)
    return total_params, size_mb


def print_model_summary(models: dict):
    """Prints a table with model names, parameter counts, and sizes."""
    print(f"{'Model':<20} {'Params (M)':>12} {'Size (MB)':>12}")
    print("-" * 46)
    total_params = 0
    total_size = 0
    for name, model in models.items():
        params, size = count_parameters(model)
        print(f"{name:<20} {params/1e6:>12.3f} {size:>12.2f}")
        total_params += params
        total_size += size
    print("-" * 46)
    print(f"{'TOTAL':<20} {total_params/1e6:>12.3f} {total_size:>12.2f} MB\n")

def get_base_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


if __name__ == "__main__":
    rospy.init_node('Gazebo_test', anonymous=True)
    pose_listener = OdomPoseListener("/odom")
    print("Subscribed to /odom for pose tracking")
    rospy.Subscriber("/channels", Float32MultiArray, channel_callback, queue_size=10)

    rospy.Subscriber("/clock", Clock, clock_callback)
    rospy.Subscriber("/render_done", Int32, render_done_cb)

    # rospy.Subscriber("/cam_front/world_cam/image_raw", Image, image_callback1)
    # rospy.Subscriber("/cam_back/world_cam/image_raw", Image, image_callback2)
    # rospy.Subscriber("/cam_left/world_cam/image_raw", Image, image_callback3)
    # rospy.Subscriber("/cam_right/world_cam/image_raw", Image, image_callback4)
    # rospy.Subscriber("/cam_top/world_cam/image_raw", Image, image_callback5)

    flag_pub = rospy.Publisher('/render_trigger', Int32, queue_size=10)

    device = "cpu"
    # model_path = "path/to/saved_model.pth"
    model_path = model_path("results", "CarRacing-v2_0_pomdp", "20_dec_gazebo", "models_best_8.pth")  #31_oct_gym  7_nov_Gazebo
    wmodel_path = model_path("wireless_models", "3_bs", "wi-jepa_")
    vae_path = model_path("masked_data", "trained_model", "50", "834", "patch_model.pt")   #old_trained_model/without_fix/patch_model.pt"     #trained_model/500/1502/patch_model.pt  #old_trained_model 
    env = GazeboEnv()

    vae_model = VAE().to(device)
    vae_model.load_state_dict(torch.load(vae_path, map_location=device))
    vae_model.eval()

    env = ImageEnv(env, skip_frames=3, stack_frames=4, initial_no_op=5)
    env = OneHotAction(env)
    # env = OneHotAction(ImageEnv(env))
    obs, _ = env.reset()
    # # Loading control model
    config = RacingCarConfig(capacity=1)
    # trainer = Trainer(config, torch.device("cpu"))
    while(channels is None):
        print("Connecting with Sionna...")
    print( "Evaluating saved agent...")
    average_score = eval_saved_agent(env, config, model_path,wmodel_path,device,flag_pub,vae_model)

