#!/usr/bin/env python3
# coding: utf-8

# ============================================================
# Imports
# ============================================================

import numpy as np
import gymnasium as gym
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt

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

rospy.init_node("dqn_gazebo_training", anonymous=True)
# ============================================================
# Config
# ============================================================

SAVE_DIR = "training/dqn"
os.makedirs(SAVE_DIR, exist_ok=True)

PRETRAINED_MODEL = "training/dqn/dqn_best.pt"

wandb.init(
    project="dqn-carracing",
    name="DQN-Gazebo-Transfer",
)

# ============================================================
# Preprocessing
# ============================================================

# def preprocess(img):
#     print(img.shape)
#     img = cv2.resize(img, (84, 84))
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
#     return img

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

    # If RGB -> grayscale
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = img.astype(np.float32) / 255.0
    return img


class ImageEnv(gym.Wrapper):
    def __init__(self, env, skip_frames=3, stack_frames=4, initial_no_op=5):
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
# CNN Q Network
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

# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, state_dim, max_size=int(1e5)):
        self.s = np.zeros((max_size, *state_dim), np.float32)
        self.a = np.zeros((max_size, 1), np.int64)
        self.r = np.zeros((max_size, 1), np.float32)
        self.s2 = np.zeros((max_size, *state_dim), np.float32)
        self.d = np.zeros((max_size, 1), np.float32)
        self.ptr, self.size, self.max_size = 0, 0, max_size

    def add(self, s, a, r, s2, d):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s2[self.ptr] = s2
        self.d[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch):
        idx = np.random.randint(0, self.size, batch)
        return (
            torch.FloatTensor(self.s[idx]),
            torch.LongTensor(self.a[idx]),
            torch.FloatTensor(self.r[idx]),
            torch.FloatTensor(self.s2[idx]),
            torch.FloatTensor(self.d[idx]),
        )

# ============================================================
# DQN Agent
# ============================================================

class DQN:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q = CNNActionValue(state_dim[0], action_dim).to(self.device)
        self.q_tgt = CNNActionValue(state_dim[0], action_dim).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())

        self.opt = torch.optim.RMSprop(
            self.q.parameters(), lr=2.5e-4, alpha=0.95, eps=0.01
        )

        self.buffer = ReplayBuffer(state_dim)
        self.gamma = 0.99
        self.batch = 32

        self.eps = 0.3              # LOWER epsilon for transfer learning
        self.eps_min = 0.05
        self.eps_decay = (self.eps - self.eps_min) / 5e5

        self.warmup = 2000
        self.sync = 10000
        self.steps = 0

    def act(self, s, train=True):
        if train and (np.random.rand() < self.eps or self.steps < self.warmup):
            return np.random.randint(env.action_space.n)
        s = torch.FloatTensor(s).unsqueeze(0).to(self.device)
        return self.q(s).argmax().item()

    def update(self):
        s, a, r, s2, d = [x.to(self.device) for x in self.buffer.sample(self.batch)]
        with torch.no_grad():
            y = r + (1 - d) * self.gamma * self.q_tgt(s2).max(1, keepdim=True)[0]
        q = self.q(s).gather(1, a)
        loss = F.smooth_l1_loss(q, y)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        wandb.log({
            "train/loss": loss.item(),
            "train/epsilon": self.eps,
            "step": self.steps,
        })

    def step_update(self, data):
        self.steps += 1
        self.buffer.add(*data)

        if self.steps > self.warmup:
            self.update()

        if self.steps % self.sync == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())

        self.eps = max(self.eps_min, self.eps - self.eps_decay)

# ============================================================
# Training
# ============================================================

env = ImageEnv(
    GazeboEnv()
)

agent = DQN((4, 84, 84), env.action_space.n)

# ---- Load pretrained Gym model ----
if os.path.exists(PRETRAINED_MODEL):
    agent.q.load_state_dict(torch.load(PRETRAINED_MODEL, map_location=agent.device))
    agent.q_tgt.load_state_dict(agent.q.state_dict())
    print("✅ Loaded pretrained Gym DQN model")
else:
    raise FileNotFoundError("❌ Pretrained Gym model not found")

# ============================================================
# Training Loop
# ============================================================

max_steps = int(1e6)
eval_interval = 10000

best_return = -1e9

episode_reward = 0.0
episode_count = 0
episode_rewards = []

s, _ = env.reset()

while agent.steps < max_steps:
    a = agent.act(s)
    s2, r, t, tr, _ = env.step(a)

    agent.step_update((s, a, r, s2, t))
    s = s2
    episode_reward += r

    if t or tr:
        episode_rewards.append(episode_reward)

        wandb.log({
            "gazebo/episode_reward": episode_reward,
            "gazebo/episode": episode_count,
            "step": agent.steps,
        })

        print(f"[EPISODE {episode_count}] reward = {episode_reward:.2f}")

        episode_reward = 0.0
        episode_count += 1
        s, _ = env.reset()

    if agent.steps % eval_interval == 0:
        avg = np.mean(episode_rewards[-5:]) if len(episode_rewards) >= 5 else np.mean(episode_rewards)

        print(f"[EVAL] step={agent.steps} avg_return={avg:.2f}")

        ckpt = f"{SAVE_DIR}/gazebo_step_{agent.steps}.pt"
        torch.save(agent.q.state_dict(), ckpt)
        wandb.save(ckpt)

        if avg > best_return:
            best_return = avg
            best_path = f"{SAVE_DIR}/gazebo_best.pt"
            torch.save(agent.q.state_dict(), best_path)
            wandb.save(best_path)
            print(f"[BEST] New best avg_return={best_return:.2f}")

wandb.finish()


