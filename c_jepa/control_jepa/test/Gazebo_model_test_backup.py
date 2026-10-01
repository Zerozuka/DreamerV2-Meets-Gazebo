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
# paths.py は gz_sionna/src に置いて全パッケージで共有している。
# gz_sionna はまだ Python モジュールを install していないので share のパスを通す。
import os as _p_os
import sys as _p_sys
from ament_index_python.packages import get_package_share_directory as _p_share
_p_sys.path.insert(0, _p_os.path.join(_p_share('gz_sionna'), 'src'))
from ros_image import imgmsg_to_bgr8  # noqa: E402
from paths import model_path, output_path


csv_path = output_path("predicted_power_log.csv")
case_id = "case_1/"
output_dir = output_path("Proposed") + case_id
datapath = output_dir + "proposed_results.pt"
_video_writers = {} 
last_completed = None

prev_frame_global1 = None
prev_frame_global2 = None
prev_frame_global3 = None
prev_frame_global4 = None
prev_frame_global5 = None

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





def eval_saved_agent(env, config, model_path,wmodel_path ,device,flag_pub):
    global channels,last_completed
    RSSMModel, ObsEncoderModel, ActionModel = load_model(config, model_path, device)
    encoder , predictor , power_predictor = load_wireless_model(wmodel_path, device)
    eval_episode = config.eval_episode
    eval_scores = []
    action_size = config.action_size
    dataset = {"step": [],"poses": [], "reward": [],"channels": [],"csi_embedding": [],"latent_state": [],"action": [],"predicted_power": [],"uplink_com_status": []}
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
    horizon_len = 5

    real_start_time = time.time()
    sim_start_time = None
    init_steps = 200000
    flag_h = True
    sequence_mode = False
    sequence_index = 0
    sequence_target = 0

    main_act_horizon = None
    main_power_h = None
    main_prev_rssmstate = None
    run_jepa = True
    noise_ratio = 0.000115

    while not rospy.is_shutdown() and not done:
        # print(channels.shape)
        # print(channels.shape)
        if (last_completed is None or frame_id == 0):

            

            chan_fft = torch.fft.fft(channels)   # was: channels = torch.fft.fft(channels)
            g = chan_fft[0]
            p_flat = chan_fft.reshape(chan_fft.size(0), -1)
            actual_power = torch.linalg.vector_norm(
                p_flat.reshape(-1, 3, 8, 16)[:, 2, :, 15], dim=1
            )
            actual_power = 10 * torch.log10(actual_power)

            if sim_start_time is None and sim_time_sec > 0:
                sim_start_time = sim_time_sec
            
            with torch.no_grad():
                
                flag_pub.publish(1)
                while (last_completed!=1):
                    # print("loop")
                    af = 0

                g_best = torch.abs(g).max()
                channel_condition = (noise_ratio > g_best.item())

                if(run_jepa == True):
                    embed = ObsEncoderModel(torch.tensor(obs, dtype=torch.float32)
                                            .unsqueeze(0).to(device))
                    _, posterior_rssm_state = RSSMModel.rssm_observe(
                        embed, prev_action, not done, prev_rssmstate
                    )
                    model_state = RSSMModel.get_model_state(posterior_rssm_state)
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

                    action_horizon.append(action.squeeze(0).cpu().numpy())
                    power_pred_horizon.append(pred_power_value)
                    model_state_horizon.append(model_state.squeeze(0).cpu().numpy())
                    prev_rssmstate = posterior_rssm_state
                    prev_action = action
                    prev_rssmstate_h.append(prev_rssmstate)

                    for t in range(horizon_len-1):
                        posterior_rssm_state = RSSMModel.rssm_imagine(prev_action, prev_rssmstate)
                        model_state = RSSMModel.get_model_state(posterior_rssm_state)
                        action, _ = ActionModel(model_state)

                        latent_dynamics = model_state.unsqueeze(0)
                        predictions_, _ = predictor(latent_dynamics.float())
                        predictions += predictions_
                        predicted_power = power_predictor(predictions)
                        pred_power_value = predicted_power.squeeze().cpu().numpy()
                        
                        power_pred_horizon.append(pred_power_value)
                        action_horizon.append(action.squeeze(0).cpu().numpy())
                        model_state_horizon.append(model_state.squeeze(0).cpu().numpy())
                    
                        prev_rssmstate = posterior_rssm_state
                        prev_action = action
                        prev_rssmstate_h.append(prev_rssmstate)

                
            
                

                px, py = pose_listener.get_pose()


                if not sequence_mode:
                    # update main horizon/power
                    main_act_horizon = action_horizon
                    main_power_h = power_pred_horizon
                    main_prev_rssmstate = prev_rssmstate_h

                    if frame_id<init_steps:
                        print(f"[Frame {frame_id}] Action init = {action_horizon[0]}")
                        next_obs, rew, terminated, truncated, info = env.step(action_horizon[0])
                        dataset["step"].append(frame_id)
                        dataset["poses"].append([float(px), float(py)])
                        dataset["reward"].append(rew)
                        dataset["channels"].append(chan_fft)
                        dataset["csi_embedding"].append(csi_embed.squeeze(0).cpu().numpy())
                        dataset["latent_state"].append(model_state_horizon)
                        dataset["action"].append(action_horizon)
                        dataset["predicted_power"].append(power_pred_horizon)
                        dataset["uplink_com_status"].append(int(1))

                        prev_rssmstate = prev_rssmstate_h[0]
                        prev_action = torch.tensor(action_horizon[0], dtype=torch.float32).unsqueeze(0)
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



                if(flag_h==False):  # Horizon mode
                    
                    temp_act_h = action_horizon
                    temp_power = power_pred_horizon
                    print(f"[Frame {frame_id}] Action  = {main_act_horizon[sequence_index]}")
                    next_obs, rew, terminated, truncated, info = env.step(main_act_horizon[sequence_index])
                    dataset["step"].append(frame_id)
                    dataset["poses"].append([float(px), float(py)])
                    dataset["reward"].append(rew)
                    dataset["channels"].append(chan_fft)
                    dataset["csi_embedding"].append(csi_embed.squeeze(0).cpu().numpy())
                    dataset["latent_state"].append(model_state_horizon)
                    dataset["action"].append(action_horizon)
                    dataset["predicted_power"].append(power_pred_horizon)
                    if sequence_index ==0:
                        dataset["uplink_com_status"].append(int(1))
                    else:
                        dataset["uplink_com_status"].append(int(0))
                    sequence_index += 1

                    frame_id += 1
                    score += rew
                    obs = next_obs
                    last_completed = None

                    if sequence_index > sequence_target:
                        prev_rssmstate = main_prev_rssmstate[sequence_target] 
                        prev_action = torch.tensor(main_act_horizon[sequence_target], dtype=torch.float32).unsqueeze(0)
                        sequence_mode = False
                        flag_h = True
                        run_jepa = True

            if(truncated==True or terminated==True):
                print("loop truncated")
                done = True

        
            else:
                done = False
        

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
    model_path = model_path("results", "CarRacing-v2_0_pomdp", "20_dec_gazebo", "models_best_8.pth")   #31_oct_gym  7_nov_Gazebo
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

