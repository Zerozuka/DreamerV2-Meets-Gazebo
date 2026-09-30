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
from cv_bridge import CvBridge
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
from paths import model_path, output_path


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

bridge = CvBridge()
if os.path.exists(csv_path):
    os.remove(csv_path)

channels = None


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
    img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
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
    img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
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
    img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
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
    img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
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
    img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
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

def eval_saved_agent(env, config, model_path,wmodel_path ,device,flag_pub):
    global channels,last_completed
    RSSMModel, ObsEncoderModel, ActionModel = load_model(config, model_path, device)
    encoder , predictor , power_predictor = load_wireless_model(wmodel_path, device)
    eval_episode = config.eval_episode
    eval_scores = []
    action_size = config.action_size
    dataset = {"step": [],"image": [],"poses": [], "reward": [],"channels": [],"csi_embedding": [],"latent_state": [],"action": [],"predicted_power": [],"uplink_com_status": [],"color_area_id":[]}
    testdata = {"step": [],"z_value": [],"h_value":[],"rssm_state":[],"z_h":[]}
    # for e in range(eval_episode):
    obs, info = env.reset()
    score = 0
    done = False
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
    init_steps = 20
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


    base_env = get_base_env(env)
    base_env.set_visual_mode(0)
    csv_rows = []

    prev_probs = None
    prev_h = None
    lighting_counter = 0

    KL_THRESH = 0.3        
    H_THRESH  = 0.013333333 #0.02      
    CONFIRM_STEPS = 3

    prev_probs_ref = None
    prev_h_ref = None

    # z_mean_ref = 
    prev_z_h = None

    mse1 = torch.tensor(0.0, device="cpu")
    mse2 = torch.tensor(0.0, device="cpu")
    mse3 = torch.tensor(0.0, device="cpu")

    cos_sim1 = torch.tensor(0.0, device="cpu")
    cos_sim2 = torch.tensor(0.0, device="cpu")
    cos_sim3 = torch.tensor(0.0, device="cpu")


    mse_1 = torch.tensor(0.0, device="cpu")
    mse_2 = torch.tensor(0.0, device="cpu")
    mse_3 = torch.tensor(0.0, device="cpu")

    cos_sim_1 = torch.tensor(0.0, device="cpu")
    cos_sim_2 = torch.tensor(0.0, device="cpu")
    cos_sim_3 = torch.tensor(0.0, device="cpu")

    prev_prev_z_h = None
    prev_z = None
    color_idx = 0
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



                    embed = ObsEncoderModel(torch.tensor(obs, dtype=torch.float32)
                                            .unsqueeze(0).to(device))
                    _, posterior_rssm_state = RSSMModel.rssm_observe(
                        embed, prev_action, not done, prev_rssmstate
                    )
                    model_state = RSSMModel.get_model_state(posterior_rssm_state)
                    stoch_z = model_state[:, :1024]
                   
                    z_dim = 1024                 # confirm once if needed
                    z = model_state[:, :z_dim]      # stochastic (not used)
                    h = model_state[:, z_dim:]  
                    
                    testdata["step"].append(frame_id)
                    testdata["z_value"].append(z)
                    testdata["h_value"].append(h)
                    testdata["rssm_state"].append(posterior_rssm_state)
                    
                   
                    if(frame_id>1):

                        A = z
                        B = prev_z_h[1] 
                        # print(A.shape)
                        # print(B.shape)

                        cos_sim1 = F.cosine_similarity(A.flatten(), B.flatten(), dim=0)
                        mse1 = F.mse_loss(A, B)

                        A2 = z
                        B2 = prev_prev_z_h[2]
                        # print(A2.shape)
                        # print(B2.shape)
                        cos_sim2 = F.cosine_similarity(A2.flatten(), B2.flatten(), dim=0)
                        mse2 = F.mse_loss(A2, B2)

                        A3 = prev_prev_z_h[2]
                        B3 = prev_z_h[1]
                        # print(A2.shape)
                        # print(B2.shape)
                        cos_sim3 = F.cosine_similarity(A3.flatten(), B3.flatten(), dim=0)
                        mse3 = F.mse_loss(A3, B3)

                    
                    latent_error = 1*cos_sim1 + 0.5*cos_sim2+0.75*cos_sim3
                    latent_error2 = 1*mse1.item() + 0.5*mse2.item() + 0.75*mse3.item()
                    latent_error = (
                        1.0 * cos_sim1 +
                        1 * cos_sim2 +
                        1 * cos_sim3
                    )

                    latent_error2 = (
                        1.0 * mse1 +
                        1 * mse2 +
                        1 * mse3
                    )

                    latent_error_val = latent_error.item()
                    latent_error2_val = latent_error2.item()


                    csv_rows.append({ "id": frame_id,"latent_error": latent_error_val,"latent_error2": latent_error2_val})

                    if(frame_id>100 and latent_error2_val>0.7 and latent_error_val<1.2):
                        
                        lighting_counter += 1
                        
                    
                    else:
                        lighting_counter = 0

                    # if(lighting_counter>=1):
                    #     print("New area")
                    #     lighting_counter = 0

                        
                    #     obs_white = np.expand_dims(base_env.get_obs_with_mode(0),axis=0)
                    #     obs_dark  = np.expand_dims(base_env.get_obs_with_mode(1),axis=0)
                    #     obs_blue  = np.expand_dims(base_env.get_obs_with_mode(2),axis=0)
                    #     #
                    #     embed1 = ObsEncoderModel(torch.tensor(obs_white, dtype=torch.float32)
                    #                         .unsqueeze(0).to(device))
                    #     _, posterior_rssm_state1 = RSSMModel.rssm_observe(
                    #         embed1, prev_action, not done, prev_rssmstate
                    #     )
                    #     model_state1 = RSSMModel.get_model_state(posterior_rssm_state1)

                    #     embed2 = ObsEncoderModel(torch.tensor(obs_dark, dtype=torch.float32)
                    #                         .unsqueeze(0).to(device))
                    #     _, posterior_rssm_state2 = RSSMModel.rssm_observe(
                    #         embed2, prev_action, not done, prev_rssmstate
                    #     )
                    #     model_state2 = RSSMModel.get_model_state(posterior_rssm_state2)

                    #     embed3 = ObsEncoderModel(torch.tensor(obs_blue, dtype=torch.float32)
                    #                         .unsqueeze(0).to(device))
                    #     _, posterior_rssm_state3 = RSSMModel.rssm_observe(
                    #         embed3, prev_action, not done, prev_rssmstate
                    #     )
                    #     model_state3 = RSSMModel.get_model_state(posterior_rssm_state3)

                    #     z1 = model_state1[:, :z_dim]
                    #     z2 = model_state2[:, :z_dim]
                    #     z3 = model_state3[:, :z_dim]

                    #     A = z1
                    #     B = prev_prev_z_h[2]

                        
                        
                    #     cos_sim_1 = F.cosine_similarity(A.flatten(), B.flatten(), dim=0)
                    #     mse_1 = F.mse_loss(A, B)

                    #     A = z2
                    #     B = prev_prev_z_h[2]

                    #     cos_sim_2 = F.cosine_similarity(A.flatten(), B.flatten(), dim=0)
                    #     mse_2 = F.mse_loss(A, B)

                    #     A = z3
                    #     B = prev_prev_z_h[2]

                    #     cos_sim_3 = F.cosine_similarity(A.flatten(), B.flatten(), dim=0)
                    #     mse_3 = F.mse_loss(A, B)

                    #     cos_sims = {
                    #         "white": 0,
                    #         "dark":  cos_sim_2,
                    #         "blue":  cos_sim_3,
                    #     }

                    #     mses = {
                    #         "white": 10000,
                    #         "dark":  mse_2,
                    #         "blue":  mse_3,
                    #     }

                    #     best_mode1 = max(cos_sims, key=cos_sims.get)
                    #     best_mode2 = min(mses, key=mses.get)

                    #     best_value1 = cos_sims[best_mode1].item()
                    #     best_value2 = mses[best_mode1].item()
                        
                    #     print(best_mode1)
                    #     print(best_mode2)
                    #     print(best_value1)
                    #     print(best_value2)

                    #     if(best_mode1==best_mode2 and best_mode1=="white" and color_idx ==0):
                    #         print("changed")
                    #         base_env.set_visual_mode(0)
                    #         model_state = model_state1
                    #         posterior_rssm_state = posterior_rssm_state1
                    #         z = z1
                    #         color_idx = 0

                    #     elif(best_mode1==best_mode2 and best_mode1=="dark" and color_idx ==0 and frame_id>132): #or (frame_id>140 and frame_id<330 and color_idx ==0)):
                    #         print("changed")
                    #         base_env.set_visual_mode(1)
                    #         model_state = model_state2
                    #         posterior_rssm_state = posterior_rssm_state2
                    #         z = z2
                    #         color_idx = 1
                    #         for i in range(4):
                    #             frame_id += 1
                    #             dataset["step"].append(frame_id)
                    #             dataset["poses"].append([float(px), float(py)])
                    #             dataset["reward"].append(-0.3)
                    #             dataset["channels"].append(chan_fft)
                    #             dataset["csi_embedding"].append(csi_embed.squeeze(0).cpu().numpy())
                    #             dataset["latent_state"].append(model_state_horizon)
                    #             dataset["action"].append(action_horizon)
                    #             dataset["predicted_power"].append(power_pred_horizon)
                    #             dataset["uplink_com_status"].append(int(0))
                    #             dataset["color_area_id"].append(color_idx)


                        
                    #     elif(best_mode1==best_mode2 and best_mode1=="blue" and color_idx ==1 and frame_id>340): #or (frame_id>340 and color_idx==1)):
                    #         print("changed")
                    #         base_env.set_visual_mode(2)
                    #         model_state = model_state3
                    #         posterior_rssm_state = posterior_rssm_state3
                    #         z = z3
                    #         color_idx = 2
                    #         for i in range(4):
                    #             frame_id += 1
                    #             dataset["step"].append(frame_id)
                    #             dataset["poses"].append([float(px), float(py)])
                    #             dataset["reward"].append(-0.3)
                    #             dataset["channels"].append(chan_fft)
                    #             dataset["csi_embedding"].append(csi_embed.squeeze(0).cpu().numpy())
                    #             dataset["latent_state"].append(model_state_horizon)
                    #             dataset["action"].append(action_horizon)
                    #             dataset["predicted_power"].append(power_pred_horizon)
                    #             dataset["uplink_com_status"].append(int(0))
                    #             dataset["color_area_id"].append(color_idx)


                        # if(best_value1>0.4 and best_mode1=="white" and frame_id>500):
                        #     print("changed")
                        #     base_env.set_visual_mode(0)
                        #     model_state = model_state1
                        #     posterior_rssm_state = posterior_rssm_state1
                        #     z = z1
                        #     color_idx = 0

                        # elif(best_value1>0.2 and best_mode1=="dark" and frame_id>130 and color_idx == 0): #or (frame_id>140 and frame_id<330 and color_idx ==0)):
                        #     print("changed")
                        #     base_env.set_visual_mode(1)
                        #     model_state = model_state2
                        #     posterior_rssm_state = posterior_rssm_state2
                        #     z = z2
                        #     color_idx = 1

                        
                        # elif(best_value1>0.35 and best_mode1=="blue" and frame_id>310 and color_idx ==1): #or (frame_id>340 and color_idx==1)):
                        #     print("changed")
                        #     base_env.set_visual_mode(2)
                        #     model_state = model_state3
                        #     posterior_rssm_state = posterior_rssm_state3
                        #     z = z3
                        #     color_idx = 2



                        
                        

                    prev_z = z

                    action, _ = ActionModel(model_state)

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
                    stoch_vals = []
                    stoch_vals.append(stoch_z)
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
                        stoch_z = model_state[:, :1024]
                        stoch_vals.append(stoch_z)

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

                    testdata["z_h"].append(stoch_vals)
                    prev_prev_z_h = prev_z_h
                    prev_z_h = stoch_vals
                
                
                

               
            
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
                    
                    dataset["step"].append(frame_id)
                    dataset["poses"].append([float(px), float(py)])
                    dataset["reward"].append(rew)
                    dataset["channels"].append(chan_fft)
                    dataset["csi_embedding"].append(csi_embed.squeeze(0).cpu().numpy())
                    dataset["latent_state"].append(model_state_horizon)
                    dataset["action"].append(action_horizon)
                    dataset["predicted_power"].append(power_pred_horizon)
                    dataset["color_area_id"].append(color_idx)

                    
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

    print("Saved timing info to:", timing_path)

    torch.save(dataset, datapath)
    torch.save(testdata, datapath_2)
    print("Saved dataset with", len(dataset["poses"]), "samples")
    print("frame_id ", frame_id)
    print(f'Average evaluation score for model at = {avg_score}')
    env.close()
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
    env = GazeboEnv()

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
    average_score = eval_saved_agent(env, config, model_path,wmodel_path,device,flag_pub)

