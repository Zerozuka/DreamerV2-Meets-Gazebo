#!/usr/bin/env python3
import cv2
import gymnasium as gym
import numpy as np

# 旧 OpenAI Gym の import (import gym) は削除した。gymnasium と併存すると
# gym.Env と gymnasium.Env が別クラスになり、ラッパーの継承関係が壊れる。
# rospy も削除した。ロギングは呼び出し側の env が持つノードを使う。


def preprocess(img):
    """
    Convert Gazebo camera frame to 84x84 grayscale normalized image.
    """
    # If your Gazebo camera has a wide view, crop as needed
    # Example: crop horizon / floor regions if unnecessary
    # img = img[30:210, 60:280]  # optional cropping

    img = cv2.resize(img, (84, 84))
    cv2.imshow("img", img)
    cv2.waitKey(1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0
    return img  # shape: (84, 84)

class ImageEnv(gym.Wrapper):
    """
    Preprocesses GazeboEnv camera images:
    - Converts RGB to 84x84 grayscale
    - Optionally stacks frames
    - Skips frames to speed up learning
    - Adds 'no-op' steps at start (like in Atari)
    """

    def __init__(self, env, skip_frames=3, stack_frames=4, initial_no_op=50, **kwargs):
        super(ImageEnv, self).__init__(env)
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
        self.initial_no_op = initial_no_op
        # self.stacked_state = None

    def reset(self, *args, **kwargs):
        """
        Reset GazeboEnv, perform some initial 'no-op' steps, and stack frames.
        """
        obs, info = self.env.reset(*args, **kwargs)

        # Perform initial no-op steps (simply move forward slightly)
        print(f"Performing {self.initial_no_op} no-op steps...")
        
        for _ in range(self.initial_no_op):
            # dummy_action = np.zeros(self.env.action_space.shape[0], dtype=np.float32)
            obs, reward, terminated, truncated , info = self.env.step(0)

        # Preprocess and stack frames
        frame = preprocess(obs.transpose(1, 2, 0))  # (84,84)
        # self.stacked_state = np.stack([frame] * self.stack_frames, axis=0)  # (4,84,84)
        self.stacked_state = np.expand_dims(frame, axis=0)

        return self.stacked_state, info

    def step(self, action):
        """
        Repeats an action for `skip_frames` steps, stacks latest frame.
        """
        total_reward = 0.0
        terminated = False
        truncated = False
       
        info = {}

        for _ in range(self.skip_frames):
            obs, reward, terminated, truncated , info = self.env.step(action)
            total_reward += reward
            

            if terminated or truncated:
                break

        # Convert new frame to grayscale 84x84
        frame = preprocess(obs.transpose(1, 2, 0))

        # Update stacked frames (shift and append)
        # self.stacked_state = np.roll(self.stacked_state, shift=-1, axis=0)
        # self.stacked_state[-1] = frame
        self.stacked_state = np.expand_dims(frame, axis=0)

        return self.stacked_state,total_reward,terminated,truncated,info

class OneHotAction(gym.Wrapper):
    """
    Converts discrete integer action space → one-hot continuous Box space.
    Makes GazeboEnv compatible with DreamerV2's continuous action policy output.
    """

    def __init__(self, env):
        assert isinstance(
            env.action_space, gym.spaces.Discrete
        ), "GazeboEnv must have Discrete action space for OneHotAction"

        shape = (env.action_space.n,)
        env.action_space = gym.spaces.Box(
            low=0, high=1, shape=shape, dtype=np.float32
        )
        env.action_space.sample = self._sample_action

        super(OneHotAction, self).__init__(env)

    def step(self, action):
        """
        Convert one-hot or softmax vector → integer index → env.step(index)
        """
        index = int(np.argmax(action))
        reference = np.zeros_like(action, dtype=np.float32)
        reference[index] = 1.0
        return self.env.step(index)

    def reset(self, *args, **kwargs):
        return self.env.reset()

    def _sample_action(self):
        """
        Random one-hot sample for exploration.
        """
        action = self.env.action_space.shape[0]
        idx = np.random.randint(0, action)
        ref = np.zeros(action, dtype=np.float32)
        ref[idx] = 1.0
        return ref