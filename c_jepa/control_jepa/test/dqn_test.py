#!/usr/bin/env python3
# coding: utf-8

# ============================================================
# Imports
# ============================================================

import gymnasium as gym
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# Preprocessing (SAME AS TRAINING)
# ============================================================

# def preprocess(img):
#     img = img[:84, 6:90]
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     img = img / 255.0
#     return img.astype(np.float32)

def preprocess(img):
    img = img[:84, 6:90] # CarRacing-v2-specific cropping
    # img = cv2.resize(img, dsize=(84, 84)) # or you can simply use rescaling
    
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
        # Reset the original environment.
        s, info = self.env.reset()

        # Do nothing for the next `self.initial_no_op` steps
        for i in range(self.initial_no_op):
            s, r, terminated, truncated, info = self.env.step(0)
        
        # Convert a frame to 84 X 84 gray scale one
        s = preprocess(s)

        # The initial observation is simply a copy of the frame `s`
        self.stacked_state = np.tile(s, (self.stack_frames, 1, 1))  # [4, 84, 84]
        return self.stacked_state, info
    
    def step(self, action):
        # We take an action for self.skip_frames steps
        reward = 0
        for _ in range(self.skip_frames):
            s, r, terminated, truncated, info = self.env.step(action)
            reward += r
            if terminated or truncated:
                break

        # Convert a frame to 84 X 84 gray scale one
        s = preprocess(s)

        # Push the current frame `s` at the end of self.stacked_state
        self.stacked_state = np.concatenate((self.stacked_state[1:], s[np.newaxis]), axis=0)

        return self.stacked_state, reward, terminated, truncated, info




# ============================================================
# CNN Q Network (SAME ARCHITECTURE)
# ============================================================

class CNNActionValue(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        self.in_features = 32 * 9 * 9
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((-1, self.in_features))
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ============================================================
# DQN Agent (ONLY ACTION + LOAD)
# ============================================================

class DQNAgent:
    def __init__(self, state_dim, action_dim, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = CNNActionValue(state_dim[0], action_dim).to(self.device)
        self.network.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.network.eval()

        print(f"Model loaded from: {model_path}")
        print(f"Using device: {self.device}")

    @torch.no_grad()
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        q = self.network(state)
        return torch.argmax(q).item()


# ============================================================
# EVALUATION FUNCTION (YOUR LOGIC + FRAMES)
# ============================================================

def evaluate_and_record(
    model_path="dqn.pt",
    render=False,
    save_video=False,
    video_path="dqn_eval.mp4",
):
    env = gym.make(
        "CarRacing-v2",
        continuous=False,
        render_mode="human",
    )
    env = ImageEnv(env)

    state_dim = (4, 84, 84)
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, model_path)

    frames = []
    scores = 0.0

    state, _ = env.reset()
    done = False
    ret = 0.0

    while not done:
        if render or save_video:
            frame = env.render()
            frames.append(frame)

        action = agent.act(state)
        state, reward, terminated, truncated, _ = env.step(action)

        ret += reward
        done = terminated or truncated

    scores += ret
    env.close()

    print(f"Episode return: {scores:.2f}")

    # ========================================================
    # Save video (optional)
    # ========================================================
    if save_video:
        print(f"Saving video to {video_path}")
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (width, height),
        )
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print("Video saved successfully.")

    # ========================================================
    # Show last frame
    # ========================================================
    # if render:
    #     plt.imshow(frames[-1])
    #     plt.axis("off")
    #     plt.show()

    return scores


# ============================================================
# RUN TEST
# ============================================================

if __name__ == "__main__":
    evaluate_and_record(
        model_path="training/gazebo/dqn_50000.pt",
        render=False,
        save_video=False,
        video_path="car_racing_dqn.mp4",
    )
