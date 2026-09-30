import numpy as np
import gymnasium as gym
import torch
from dreamerv2.models.actor import DiscreteActionModel
from dreamerv2.models.rssm import RSSM
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.pixel import ObsDecoder, ObsEncoder
import matplotlib.pyplot as plt

from dreamerv2.training.config_ import RacingCarConfig
from tqdm.auto import tqdm
import pickle
import os
import cv2
from paths import model_path



# Utility function for the wrapper
def preprocess(img):
    img = img[:84, 6:90]  # CarRacing-v2-specific cropping
    # img = cv2.resize(img, dsize=(84, 84)) # or you can simply use rescaling
    cv2.imshow("img", img)
    cv2.waitKey(1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    return img


# Wrapper for the environment for the image observation
class ImageEnv(gym.Wrapper):
    def __init__(self, env, skip_frames=3, stack_frames=4, initial_no_op=50, **kwargs):
        super(ImageEnv, self).__init__(env, **kwargs)
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames

    def reset(self, *args, **kwargs):
        # Reset the original environment.
        s, info = self.env.reset(*args, **kwargs)

        # Do nothing for the next `self.initial_no_op` steps
        for i in range(self.initial_no_op):
            s, r, terminated, truncated, info = self.env.step(0)

        # Convert a frame to 84 X 84 gray scale one
        s = preprocess(s)
        self.stacked_state = np.expand_dims(s, axis=0)

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
        self.stacked_state = np.expand_dims(s, axis=0)

        return self.stacked_state, reward, terminated, truncated, info


# Wrapper for the environment for one hot actions
class OneHotAction(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(
            env.action_space, gym.spaces.Discrete
        ), "This wrapper only works with discrete action space"
        shape = (env.action_space.n,)
        env.action_space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        env.action_space.sample = self._sample_action
        super(OneHotAction, self).__init__(env)

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        return self.env.step(index)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def _sample_action(self):
        actions = self.env.action_space.shape[0]
        index = np.random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


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


def eval_saved_agent(env, config, model_path, device):
    RSSMModel, ObsEncoderModel, ActionModel = load_model(config, model_path, device)
    eval_episode = config.eval_episode
    eval_scores = []
    action_size = config.action_size

    # for e in range(eval_episode):
    obs, info = env.reset()
    score = 0
    done = False
    prev_rssmstate = RSSMModel._init_rssm_state(1)
    prev_action = torch.zeros(1, action_size).to(device)

    while not done:
        with torch.no_grad():
            embed = ObsEncoderModel(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device))
            _, posterior_rssm_state = RSSMModel.rssm_observe(embed, prev_action, not done, prev_rssmstate)
            model_state = RSSMModel.get_model_state(posterior_rssm_state)
            action, _ = ActionModel(model_state)
            prev_rssmstate = posterior_rssm_state
            prev_action = action

        # next_obs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy())
        next_obs, rew, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
        if(terminated==True or truncated==True):
            done = True
        else:
            done = False
        if config.eval_render:
            env.render()
        score += rew
        obs = next_obs

    eval_scores.append(score)

    avg_score = np.mean(eval_scores)
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

    device = "cuda"
    # model_path = "path/to/saved_model.pth"
    model_path = model_path("results", "CarRacing-v2_0_pomdp", "gym_31_oct", "models_1980000.pth")

    env = gym.make("CarRacing-v2", continuous=False, render_mode="human")
    env = OneHotAction(ImageEnv(env))

    # # Loading control model
    config = RacingCarConfig(capacity=1)
    # trainer = Trainer(config, torch.device("cpu"))

    average_score = eval_saved_agent(env, config, model_path, device)

