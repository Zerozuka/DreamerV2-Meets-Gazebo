#!/usr/bin/env python3
# coding: utf-8

# ============================================================
# Imports
# ============================================================

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython.display import clear_output

# >>> WANDB ADD
import wandb
# <<< WANDB ADD

SAVE_DIR = "training/dqn"
os.makedirs(SAVE_DIR, exist_ok=True)

# >>> WANDB ADD
wandb.init(
    project="dqn-carracing",
    name="DQN-CarRacing-v2",
    config={
        "env": "CarRacing-v2",
        "algorithm": "DQN",
        "lr": 0.00025,
        "gamma": 0.99,
        "batch_size": 32,
        "buffer_size": int(1e5),
        "target_update_interval": 10000,
        "epsilon_start": 1.0,
        "epsilon_min": 0.1,
        "warmup_steps": 5000,
    }
)
# <<< WANDB ADD

# ============================================================
# Environment Initialization (raw)
# ============================================================

env = gym.make("CarRacing-v2", continuous=False)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

state, info = env.reset()
print("Raw state shape:", state.shape)

plt.imshow(state)
plt.axis("off")
plt.show()

# ============================================================
# Preprocessing
# ============================================================

def preprocess(img):
    img = img[:84, 6:90]  # CarRacing-v2-specific cropping
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    return img

# ============================================================
# Image Wrapper (SAME AS TRAINING)
# ============================================================

class ImageEnv(gym.Wrapper):
    def __init__(
        self,
        env,
        skip_frames=4,
        stack_frames=4,
        initial_no_op=50,
        **kwargs
    ):
        super(ImageEnv, self).__init__(env, **kwargs)
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames

    def reset(self):
        s, info = self.env.reset()

        for i in range(self.initial_no_op):
            s, r, terminated, truncated, info = self.env.step(0)

        s = preprocess(s)
        self.stacked_state = np.tile(s, (self.stack_frames, 1, 1))
        return self.stacked_state, info

    def step(self, action):
        reward = 0
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
    def __init__(self, state_dim, action_dim):
        super(CNNActionValue, self).__init__()

        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        self.in_features = 32 * 9 * 9
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((-1, self.in_features))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.s = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.a = np.zeros((max_size, *action_dim), dtype=np.int64)
        self.r = np.zeros((max_size, 1), dtype=np.float32)
        self.s_prime = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.terminated = np.zeros((max_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def update(self, s, a, r, s_prime, terminated):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_prime[self.ptr] = s_prime
        self.terminated[self.ptr] = terminated

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.s[ind]),
            torch.FloatTensor(self.a[ind]),
            torch.FloatTensor(self.r[ind]),
            torch.FloatTensor(self.s_prime[ind]),
            torch.FloatTensor(self.terminated[ind]),
        )

# ============================================================
# DQN Agent
# ============================================================

class DQN:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=0.00025,
        epsilon=1.0,
        epsilon_min=0.1,
        gamma=0.99,
        batch_size=32,
        warmup_steps=5000,
        buffer_size=int(1e5),
        target_update_interval=10000,
    ):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval

        self.network = CNNActionValue(state_dim[0], action_dim)
        self.target_network = CNNActionValue(state_dim[0], action_dim)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr)

        self.buffer = ReplayBuffer(state_dim, (1,), buffer_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        self.target_network.to(self.device)

        self.total_steps = 0
        self.epsilon_decay = (epsilon - epsilon_min) / 1e6

    @torch.no_grad()
    def act(self, x, training=True):
        self.network.train(training)

        if training and (
            np.random.rand() < self.epsilon or self.total_steps < self.warmup_steps
        ):
            return np.random.randint(0, self.action_dim)

        x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
        q = self.network(x)
        return torch.argmax(q).item()

    def learn(self):
        s, a, r, s_prime, terminated = map(
            lambda x: x.to(self.device),
            self.buffer.sample(self.batch_size),
        )

        next_q = self.target_network(s_prime).detach()
        td_target = r + (1.0 - terminated) * self.gamma * next_q.max(
            dim=1, keepdim=True
        ).values

        loss = F.mse_loss(self.network(s).gather(1, a.long()), td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(
            f"[TRAIN] Step: {self.total_steps} | "
            f"Epsilon: {self.epsilon:.4f} | "
            f"Loss: {loss.item():.6f}"
        )

        # >>> WANDB ADD
        wandb.log({
            "train/loss": loss.item(),
            "train/epsilon": self.epsilon,
            "train/step": self.total_steps,
        })
        # <<< WANDB ADD

        return {
            "total_steps": self.total_steps,
            "value_loss": loss.item(),
        }

    def process(self, transition):
        self.total_steps += 1
        self.buffer.update(*transition)

        result = {}
        if self.total_steps > self.warmup_steps:
            result = self.learn()

        if self.total_steps % self.target_update_interval == 0:
            print(f"[SYNC] Target network updated at step {self.total_steps}")
            self.target_network.load_state_dict(self.network.state_dict())

        self.epsilon -= self.epsilon_decay
        return result

# ============================================================
# Evaluation
# ============================================================

def evaluate(agent, n_evals=5):
    eval_env = ImageEnv(gym.make("CarRacing-v2", continuous=False))

    scores = 0.0
    for _ in range(n_evals):
        state, _ = eval_env.reset()
        done = False
        ret = 0.0

        while not done:
            action = agent.act(state, training=False)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            ret += reward
            done = terminated or truncated

        scores += ret

    return np.round(scores / n_evals, 4)

# ============================================================
# Training Loop
# ============================================================

env = ImageEnv(gym.make("CarRacing-v2", continuous=False))

state_dim = (4, 84, 84)
action_dim = env.action_space.n

agent = DQN(state_dim, action_dim)

max_steps = int(2e6)
eval_interval = 10000

history = {"Step": [], "AvgReturn": []}

state, _ = env.reset()

# >>> WANDB ADD
episode_reward = 0.0
episode_count = 0
# <<< WANDB ADD

while True:
    action = agent.act(state)
    next_state, reward, terminated, truncated, _ = env.step(action)

    agent.process((state, action, reward, next_state, terminated))
    state = next_state

    # >>> WANDB ADD
    episode_reward += reward
    # <<< WANDB ADD

    if terminated or truncated:
        # >>> WANDB ADD
        wandb.log({
            "train/episode_reward": episode_reward,
            "train/episode": episode_count,
            "train/step": agent.total_steps,
        })
        episode_reward = 0.0
        episode_count += 1
        # <<< WANDB ADD

        state, _ = env.reset()

    if agent.total_steps % eval_interval == 0:
        avg_return = evaluate(agent)

        # >>> WANDB ADD
        wandb.log({
            "eval/avg_return": avg_return,
            "eval/step": agent.total_steps,
        })
        # <<< WANDB ADD

        history["Step"].append(agent.total_steps)
        history["AvgReturn"].append(avg_return)

        plt.figure(figsize=(8, 5))
        plt.plot(history["Step"], history["AvgReturn"], "r-")
        plt.xlabel("Step", fontsize=14)
        plt.ylabel("AvgReturn", fontsize=14)
        plt.grid()
        plt.draw()
        plt.pause(0.001)
        plt.close()

        # >>> WANDB ADD
        ckpt_path = f"{SAVE_DIR}/dqn_{agent.total_steps}.pt"
        torch.save(agent.network.state_dict(), ckpt_path)
        wandb.save(ckpt_path)
        # <<< WANDB ADD

    if agent.total_steps > max_steps:
        break

# >>> WANDB ADD
wandb.finish()
# <<< WANDB ADD
