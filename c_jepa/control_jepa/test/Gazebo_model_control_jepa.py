import numpy as np
import gymnasium as gym
import torch
from dreamerv2.models.actor import DiscreteActionModel
from dreamerv2.models.rssm import RSSM
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.pixel import ObsDecoder, ObsEncoder
import matplotlib.pyplot as plt

from gazebo_env import GazeboEnv
from gazebo_wrappers import ImageEnv, OneHotAction

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
# paths.py は gz_sionna/src に置いて全パッケージで共有している。
# gz_sionna はまだ Python モジュールを install していないので share のパスを通す。
import os as _p_os
import sys as _p_sys
from ament_index_python.packages import get_package_share_directory as _p_share
_p_sys.path.insert(0, _p_os.path.join(_p_share('gz_sionna'), 'src'))
from paths import model_path


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


def eval_saved_agent(env, config, model_path, device):
    RSSMModel, ObsEncoderModel, ActionModel = load_model(config, model_path, device)
    eval_episode = config.eval_episode
    eval_scores = []
    action_size = config.action_size
    dataset = {"poses": [], "model_states": []}
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
    while not done:
        
        with torch.no_grad():
            embed = ObsEncoderModel(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device))
            _, posterior_rssm_state = RSSMModel.rssm_observe(embed, prev_action, not done, prev_rssmstate)
            model_state = RSSMModel.get_model_state(posterior_rssm_state)
            action, _ = ActionModel(model_state)
            
            print("Action: ", action.squeeze(0).cpu().numpy())
            h_t = posterior_rssm_state.deter.cpu().numpy().flatten()
            z_t = posterior_rssm_state.stoch.cpu().numpy().flatten()

            prv_h_t = prev_rssmstate.deter.cpu().numpy().flatten()
            prv_z_t = prev_rssmstate.stoch.cpu().numpy().flatten()
            print(model_state.shape)

            px, py = pose_listener.get_pose()

            dataset["poses"].append([float(px), float(py)])
            dataset["model_states"].append(
                model_state.squeeze(0).cpu().numpy().tolist()
            )

            
            log_data.append({
            "frame_id": frame_id,
            "prv_action": prev_action.squeeze(0).cpu().numpy().tolist(),
            "embed_mean": embed.mean().item(),
            "prv_h_mean": np.mean(prv_h_t),
            "prv_h_std": np.std(prv_h_t),
            "prv_z_mean": np.mean(prv_z_t),
            "prv_z_std": np.std(prv_z_t),
            "reward": float(rew),
            "done": bool(done),
            "action": action.squeeze(0).cpu().numpy().tolist(),
            "h_mean": np.mean(h_t),
            "h_std": np.std(h_t),
            "z_mean": np.mean(z_t),
            "z_std": np.std(z_t),
            })

            prev_rssmstate = posterior_rssm_state
            prev_action = action
        
        action_idx = int(np.argmax(action.squeeze(0).cpu().numpy()))
        if(action_idx==3):
            print("Gas")
        next_obs, rew, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
        
        if(truncated==True):
            print("loop truncated")
            done = True
        elif(terminated==True):
            # done = True
            steps += 1
            obs, info = env.reset()
            print("loop count ", steps)
            
        else:
            done = False

        # if config.eval_render:
        #     env.render()
        
       
        if(steps==30):
            done=True
            print("loop terminated")

        frame_id += 1
        score += rew
        obs = next_obs


    df = pd.DataFrame(log_data)
    df.to_csv("Gazebo_log_2.csv", index=False)

    eval_scores.append(score)

    avg_score = np.mean(eval_scores)

    torch.save(dataset, "gazebo_modelstate_dataset.pt")
    print("Saved dataset with", len(dataset["poses"]), "samples")
    print("frame_id ", frame_id)
    print("loop count ", steps)
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

    device = "cpu"
    # model_path = "path/to/saved_model.pth"
    model_path = model_path("results", "CarRacing-v2_0_pomdp", "7_nov_Gazebo", "models_best_4.pth")  #31_oct_gym  7_nov_Gazebo
  
    env = GazeboEnv()

    env = ImageEnv(env, skip_frames=3, stack_frames=4, initial_no_op=5)
    env = OneHotAction(env)
    # env = OneHotAction(ImageEnv(env))
    obs, _ = env.reset()
    # # Loading control model
    config = RacingCarConfig(capacity=1)
    # trainer = Trainer(config, torch.device("cpu"))
    print( "Evaluating saved agent...")
    average_score = eval_saved_agent(env, config, model_path, device)

