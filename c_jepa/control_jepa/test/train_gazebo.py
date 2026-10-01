import gymnasium as gym
import numpy as np
import torch
from dreamerv2.training.config_ import RacingCarConfig
from dreamerv2.training.config import MinAtarConfig
from dreamerv2.training.trainer import Trainer
from dreamerv2.training.evaluator import Evaluator
import wandb
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pickle
import os
import cv2
import argparse
import minatar
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

from gazebo_env import GazeboEnv
from gazebo_wrappers import ImageEnv, OneHotAction
# paths.py は gz_sionna/src に置いて全パッケージで共有している。
# gz_sionna はまだ Python モジュールを install していないので share のパスを通す。
import os as _p_os
import sys as _p_sys
from ament_index_python.packages import get_package_share_directory as _p_share
_p_sys.path.insert(0, _p_os.path.join(_p_share('gz_sionna'), 'src'))
from paths import model_path


# import tensorflow as tf
# import sionna


# # Utility function for the wrapper
# def preprocess(img):
#     img = img[:84, 6:90]  # CarRacing-v2-specific cropping
#     cv2.imshow("img", img)
#     # cv2.waitKey(1)
#     # img = cv2.resize(img, dsize=(84, 84)) # or you can simply use rescaling
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
#     # cv2.imshow("img_", img)
#     cv2.waitKey(1)
#     return img.astype(np.float32) #return img


  
# # Wrapper for the environment for the image observation
# class ImageEnv(gym.Wrapper):
#     def __init__(self, env, skip_frames=3, stack_frames=4, initial_no_op=50, **kwargs):
#         super(ImageEnv, self).__init__(env, **kwargs)
#         self.initial_no_op = initial_no_op
#         self.skip_frames = skip_frames
#         self.stack_frames = stack_frames

#     def reset(self, *args, **kwargs):
#         # Reset the original environment.
#         s, info = self.env.reset(*args, **kwargs)

#         # Do nothing for the next `self.initial_no_op` steps
#         for i in range(self.initial_no_op):
#             s, r, terminated, truncated, info = self.env.step(0)

#         # Convert a frame to 84 X 84 gray scale one
#         s = preprocess(s)
#         self.stacked_state = np.expand_dims(s, axis=0)

#         return self.stacked_state, info

#     def step(self, action):
#         # We take an action for self.skip_frames steps
#         reward = 0
#         for _ in range(self.skip_frames):
#             s, r, terminated, truncated, info = self.env.step(action)
#             reward += r
#             if terminated or truncated:
#                 break

#         # Convert a frame to 84 X 84 gray scale one
#         s = preprocess(s)

#         # Push the current frame `s` at the end of self.stacked_state
#         self.stacked_state = np.expand_dims(s, axis=0)

#         return self.stacked_state, reward, terminated, truncated, info


# class OneHotAction(gym.Wrapper):
#     def __init__(self, env):
#         assert isinstance(
#             env.action_space, gym.spaces.Discrete
#         ), "This wrapper only works with discrete action space"
#         shape = (env.action_space.n,)
#         env.action_space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
#         env.action_space.sample = self._sample_action
#         super(OneHotAction, self).__init__(env)

#     def step(self, action):
#         index = np.argmax(action).astype(int)
#         reference = np.zeros_like(action)
#         reference[index] = 1
#         return self.env.step(index)

#     def reset(self, *args, **kwargs):
#         return self.env.reset(*args, **kwargs)

#     def _sample_action(self):
#         actions = self.env.action_space.shape[0]
#         index = np.random.randint(0, actions)
#         reference = np.zeros(actions, dtype=np.float32)
#         reference[index] = 1.0
#         return reference

def main(args):
    wandb.login()
    # --- Hyperparameters ---
    # init_steps = 5
    # plan_horizon = 10
    # trust_steps = 1
    # n_sampled_states = 3

    env_name = "CarRacing-v2"
    exp_id = args.id + '_pomdp'

    '''make dir for saving results'''
    result_dir = os.path.join('results', '{}_{}'.format(env_name, exp_id))
    model_dir = os.path.join(result_dir, 'models')                                                  #dir to save learnt models
    os.makedirs(model_dir, exist_ok=True)

    env = GazeboEnv()

    env = ImageEnv(env, skip_frames=3, stack_frames=4, initial_no_op=5)
    env = OneHotAction(env)
    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
    print("Using device:", device)

    # --- Observation / Action sizes ---
    obs, _ = env.reset()
    # if len(obs.shape) == 3 and obs.shape[0] == 4:
    #     obs = np.transpose(obs, (1, 2, 0))  # (H, W, C)

    obs_shape = obs.shape
    action_size = env.action_space.shape[0]
    obs_dtype = np.float32
    action_dtype = np.float32
    batch_size = args.batch_size
    seq_len = 50  # set a default sequence length
    # obs, _ = env.reset()
    # plt.imshow(obs[-1], cmap='gray')
    # plt.title("Most recent frame (84x84 grayscale)")
    # plt.show()  RacingCarConfig MinAtarConfig
    config = RacingCarConfig(
        env=env_name,
        obs_shape=obs_shape,
        action_size=action_size,
        obs_dtype = obs_dtype,
        action_dtype = action_dtype,
        seq_len = seq_len,
        batch_size = batch_size,
        model_dir=model_dir, 
    )

    print("Observation shape:", obs_shape)
    print("Action size:", action_size)
    print("Batch size:", batch_size)
    print("Sequence length:", seq_len)
    # print("Config dict:", config.__dict__)

    config_dict = config.__dict__
    trainer = Trainer(config, device)
    evaluator = Evaluator(config, device)

    pretrained_path = model_path("results", "CarRacing-v2_0_pomdp", "models", "models_best_based.pth")
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}")
        saved_dict = torch.load(pretrained_path, map_location=device)
        trainer.load_save_dict(saved_dict)
        print("✅ Pretrained model loaded successfully!\nContinuing training in Gazebo environment...\n")
    else:
        print("⚠️ No pretrained model found at", pretrained_path)



    with wandb.init(project='Gazebo_Env', config=config_dict):
        """training loop"""
        print('...training...')
        train_metrics = {}
        trainer.collect_seed_episodes(env)
        obs, info = env.reset()
        score = 0
        done = False
        prev_rssmstate = trainer.RSSM._init_rssm_state(1)
        prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
        episode_actor_ent = []
        scores = []
        best_mean_score = 0
        train_episodes = 0
        best_save_path = os.path.join(model_dir, 'models_best.pth')

        for iter in range(1, trainer.config.train_steps):  
            if iter%trainer.config.train_every == 0:
                train_metrics = trainer.train_batch(train_metrics)
            if iter%trainer.config.slow_target_update == 0:
                trainer.update_target()                
            if iter%trainer.config.save_every == 0:
                trainer.save_model(iter)
            with torch.no_grad():
                embed = trainer.ObsEncoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(trainer.device))  
                _, posterior_rssm_state = trainer.RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                model_state = trainer.RSSM.get_model_state(posterior_rssm_state)
                action, action_dist = trainer.ActionModel(model_state)
                action = trainer.ActionModel.add_exploration(action, iter).detach()
                action_ent = torch.mean(action_dist.entropy()).item()
                episode_actor_ent.append(action_ent)

            # next_obs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy())
            next_obs, rew, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())

            score += rew

            
            print(f"action: {action.squeeze(0).cpu().numpy()}, iter: {iter} ,rew: {rew} ,score: {score} ")
            if(terminated == True or truncated == True):
                done = True
                print(terminated)
                print(truncated)
                print("***************")

            else:
                done = False

            if done:
                train_episodes += 1
                trainer.buffer.add(obs, action.squeeze(0).cpu().numpy(), rew, done)
                train_metrics['train_rewards'] = score
                train_metrics['action_ent'] =  np.mean(episode_actor_ent)
                train_metrics['train_steps'] = iter
                wandb.log(train_metrics, step=train_episodes)
                scores.append(score)
                if len(scores)>100:
                    scores.pop(0)
                    current_average = np.mean(scores)
                    if current_average>best_mean_score:
                        best_mean_score = current_average 
                        print('saving best model with mean score : ', best_mean_score)
                        save_dict = trainer.get_save_dict()
                        torch.save(save_dict, best_save_path)
                
                # obs, score = env.reset(), 0
                obs, info = env.reset()
                score = 0
                done = False
                prev_rssmstate = trainer.RSSM._init_rssm_state(1)
                prev_action = torch.zeros(1, trainer.action_size).to(trainer.device)
                episode_actor_ent = []
            else:
                trainer.buffer.add(obs, action.squeeze(0).detach().cpu().numpy(), rew, done)
                obs = next_obs
                prev_rssmstate = posterior_rssm_state
                prev_action = action

    '''evaluating probably best model'''
    # evaluator.eval_saved_agent(env, best_save_path)


if __name__ == "__main__":
    rospy.init_node('trainning_node', anonymous=True)
    parser = argparse.ArgumentParser()
    # parser.add_argument("--env", type=str, help='mini atari env name')
    parser.add_argument("--id", type=str, default='0', help='Experiment ID')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--device', default='cuda', help='CUDA or CPU')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=10, help='Sequence Length (chunk length)')
    args = parser.parse_args()
    main(args)
