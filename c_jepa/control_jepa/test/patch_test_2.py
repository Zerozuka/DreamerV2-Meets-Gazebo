import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from dreamerv2.models.actor import DiscreteActionModel
from dreamerv2.models.rssm import RSSM
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.pixel import ObsDecoder, ObsEncoder
import csv
from gazebo_env import GazeboEnv
from gazebo_wrappers import ImageEnv, OneHotAction
from cv_bridge import CvBridge
import time
from dreamerv2.training.config_ import RacingCarConfig
from tqdm.auto import tqdm
import torch.distributions as dist
from pathlib import Path
# paths.py は gz_sionna/src に置いて全パッケージで共有している。
# gz_sionna はまだ Python モジュールを install していないので share のパスを通す。
import os as _p_os
import sys as _p_sys
from ament_index_python.packages import get_package_share_directory as _p_share
_p_sys.path.insert(0, _p_os.path.join(_p_share('gz_sionna'), 'src'))
from paths import model_path

# ======================================================
# CONFIG
# ======================================================
TEST_DATA_PATH = model_path("masked_data", "proposed_results.pt")

MODEL_PATH = model_path("masked_data", "trained_model", "without_fix", "patch_model.pt") #vae_mud_denoise  patch_model_3
# DATA_PATH = model_path("masked_data", "proposed_results.pt")
BATCH_SIZE = 64
EPOCHS = 500
LR = 1e-3
LATENT_DIM = 128
VAL_RATIO = 0.1
seed = 42

MUD_COLOR = (14, 34, 49)  # BGR (OpenCV)

BASE_MODEL_DIR = Path(
    model_path("masked_data", "trained_model")
)
PATCH_MODEL_PATHS = sorted(BASE_MODEL_DIR.rglob("patch_model.pt"))
print(f"[INFO] Found {len(PATCH_MODEL_PATHS)} patch models")


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


# ======================================================
# MUD PATCH AUGMENTATION
# ======================================================
# def add_mud_patch(image):
#     """
#     image: torch.Tensor [3,84,84] or numpy [84,84,3]
#     returns: torch.Tensor [3,84,84]
#     """

#     # ---- to numpy HWC uint8 ----
#     if isinstance(image, torch.Tensor):
#         img = (image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
#     else:
#         img = image.copy()
#         if img.dtype != np.uint8:
#             img = (img * 255).astype("uint8")

#     h, w, _ = img.shape
#     overlay = img.copy()

#     num_patches = random.randint(5, 7)
#     for _ in range(num_patches):
#         cx = random.randint(0, w - 1)
#         cy = random.randint(0, h - 1)
#         r  = random.randint(10, 15)
#         cv2.circle(overlay, (cx, cy), r, MUD_COLOR, -1)

#     alpha = 1.0  # hard mud (can change to 0.6 if needed)
#     muddy = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

#     # ---- back to torch CHW float ----
#     muddy = torch.from_numpy(muddy).permute(2, 0, 1).float() / 255.0
#     return muddy

def normalized_mse(pred, target, eps=1e-8):
    """
    pred, target: [B,3,84,84] tensors in [0,1]
    returns: scalar NMSE
    """
    num = torch.sum((pred - target) ** 2)
    denom = torch.sum(target ** 2) + eps
    return num / denom

def compute_cdf(data):
    data = np.asarray(data).flatten()   # ensure 1D
    data = np.sort(data)
    cdf = np.arange(1, len(data) + 1) / len(data)
    return data, cdf

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


# ======================================================
# DATASET
# ======================================================
class MudDenoiseDataset(Dataset):
    def __init__(self, pt_path):
        data = torch.load(pt_path)
        self.images = data["srcimage"]  # numpy (84,84,3)
        self.prev_action = data["prev_action"]
        self.prev_rssmstate_ = data["prev_rssmstate"]
        self.model_state_ = data["model_state"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        clean = self.images[idx]
        prev_act = self.prev_action[idx]
        prev_rssm = self.prev_rssmstate_[idx]
        model_state = self.model_state_[idx]
        # numpy -> torch
        if isinstance(clean, np.ndarray):
            clean = torch.from_numpy(clean).permute(2, 0, 1).float()
            if clean.max() > 1:
                clean = clean / 255.0

        muddy = add_mud_patch(clean)
        return muddy, clean,prev_act,prev_rssm,model_state

# ======================================================
# VAE MODEL
# ======================================================
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # 84 → 42
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # 42 → 21
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),# 21 → 10
            nn.ReLU(),
        )

        # dynamic shape
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 84, 84)
            h = self.encoder(dummy)
            self.enc_shape = h.shape[1:]      # (128,10,10)
            self.flat_dim = h.numel()          # 12800

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # 10 → 20
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 20 → 40
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),   # 40 → 80
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

        # 🔥 recommended resize to 84×84
        recon = F.interpolate(
            recon, size=(84, 84),
            mode="bilinear",
            align_corners=False
        )

        return recon, mu, logvar


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


def transform_image_cv2_2(img, min_road_width=13, close_kernel=9, open_kernel=5, add_grass=False):
    """
    Clean Gazebo camera frame → stylized Gym-like view:
    - Road → gray (102,102,102)
    - Background → green (102,204,102)
    - Removes trees, shadows, barriers using color filtering + morphology.
    - Draws a simple car body overlay.
    """

    if img is None:
        raise FileNotFoundError("Empty frame in transform_image_cv2")

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # --- Step 1: Detect dark/gray road ---
    lower_road = np.array([0, 0, 20])
    upper_road = np.array([180, 161, 183])
    # lower_road = np.array([0, 0, 44])
    # upper_road = np.array([180, 57, 220])
    mask_road = cv2.inRange(hsv, lower_road, upper_road)

    # --- Step 2: Suppress green & sky ---
    lower_green = np.array([30, 60, 40])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_road[mask_green > 0] = 0  # remove green areas from road mask

    # --- Step 3: Morphological cleaning ---
    close_k = np.ones((close_kernel, close_kernel), np.uint8)
    open_k = np.ones((open_kernel, open_kernel), np.uint8)
    mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_CLOSE, close_k)
    mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_OPEN, open_k)

    # --- Step 4: Strengthen the road area ---
    mask_road = cv2.dilate(mask_road, np.ones((min_road_width, min_road_width), np.uint8))
    road_pixel_count = cv2.countNonZero(mask_road)
    # print("White pixel",road_pixel_count)
    # --- Step 5: Compose clean output ---
    new_img = np.full_like(img, (100, 202, 100))  # background green

    if add_grass:
        add_random_grass_patches(new_img, patch_size=150, patch_count=3)

    # Paint road gray
    new_img[mask_road > 0] = (102, 102, 102)

    # --- Step 6: Add car representation ---
    draw_gym_car_shape(new_img, center=(320, 410), scale=0.25)

    return new_img

def patched_transform(img):
    """
    Accepts:
    - torch.Tensor [3,H,W] or [H,W] or [H,W,3]
    - numpy array with same shapes
    Returns:
    - grayscale float32 [84,84] in [0,1]
    """
    # ---- torch -> numpy ----
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    # ---- handle CHW -> HWC ----
    if img.ndim == 3 and img.shape[0] in (1, 3):          # CHW
        img = np.transpose(img, (1, 2, 0))                # HWC

    # ---- if grayscale HxW -> make it 3ch BGR ----
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # ---- if float [0,1] -> uint8 ----
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    # now img is HWC uint8 (assumed BGR)
    img = cv2.resize(img, (640, 480))
    cv2.imshow("resized",img)
    img = transform_image_cv2_2(img)   # expects BGR HWC

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (84, 84))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = img.transpose(1, 2, 0)
    img = cv2.resize(img, (84, 84))
    cv2.imshow("gym_img",img)
    cv2.waitKey(1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0

    return img



def add_random_grass_patches(img, patch_size=40, patch_count=2):
        """
        Add randomized grass-colored square patches to the background,
        mimicking Gym CarRacing's textured field.
        """
        h, w, _ = img.shape
        for _ in range(patch_count):
            # Random top-left position
            x = random.randint(0, w - patch_size)
            y = random.randint(0, h - patch_size)

            # Randomized green variation
            green_variation = np.random.randint(-10, 20)
            patch_color = (
                102,
                230,
                102,
            )  # BGR (grass patch accent)

            # Draw patch
            cv2.rectangle(
                img,
                (x, y),
                (x + patch_size, y + patch_size),
                patch_color,
                thickness=-1,
            )



def draw_gym_car_shape(image, center=(320, 410), scale=0.25):
    """
    Draw a Gym CarRacing-style car with correct body and wheels.
    Args:
    image: target BGR image
    center: (x, y) center of car in image pixels
    scale: scaling factor
    """

    # ===== Original car polygons =====
    HULL_POLY1 = [(-60, +130), (+60, +130), (+60, +110), (-60, +110)]
    HULL_POLY2 = [(-15, +120), (+15, +120), (+20, +20), (-20, +20)]
    HULL_POLY3 = [
        (+25, +20),
        (+50, -10),
        (+50, -40),
        (+20, -90),
        (-20, -90),
        (-50, -40),
        (-50, -10),
        (-25, +20),
    ]
    HULL_POLY4 = [(-50, -120), (+50, -120), (+50, -90), (-50, -90)]

    polys = [HULL_POLY1, HULL_POLY2, HULL_POLY3, HULL_POLY4]

    # ===== Original wheel geometry =====
    WHEELPOS = [(-55, +80), (+55, +80), (-55, -82), (+55, -82)]
    WHEEL_R = 27
    WHEEL_W = 14

    # ===== Colors from the real Gym car =====
    HULL_COLOR = (0, 0, 202)     # deep red (BGR)
    WHEEL_COLOR = (0, 0, 0)      # black
    WHEEL_WHITE = (77, 77, 77)   # gray/white overlay

    cx, cy = center

    # ===== Draw car body =====
    for poly in polys:
        pts = np.array([[int(cx + x * scale), int(cy - y * scale)] for x, y in poly], np.int32)
        cv2.fillPoly(image, [pts], HULL_COLOR)

    # ===== Draw 4 tires =====
    for wx, wy in WHEELPOS:
        w = int(WHEEL_W * scale)
        h = int(WHEEL_R * 2 * scale)
        x = int(cx + wx * scale)
        y = int(cy - wy * scale)

        top_left = (x - w, y - h // 2)
        bottom_right = (x + w, y + h // 2)

        # Tire base
        cv2.rectangle(image, top_left, bottom_right, WHEEL_COLOR, thickness=-1)

        # Small white highlight stripe
        highlight_y1 = int(y - h * 0.3)
        highlight_y2 = int(y - h * 0.15)
        cv2.rectangle(image, (x - w, highlight_y1), (x + w, highlight_y2), WHEEL_WHITE, thickness=-1)

def transform_image_cv2(img, min_road_width=15, close_kernel=9, open_kernel=5, add_grass=False):
    """
    Clean Gazebo camera frame → stylized Gym-like view:
    - Road → gray (102,102,102)
    - Background → green (102,204,102)
    - Removes trees, shadows, barriers using color filtering + morphology.
    - Draws a simple car body overlay.
    """

    if img is None:
        raise FileNotFoundError("Empty frame in transform_image_cv2")

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # --- Step 1: Detect dark/gray road ---
    lower_road = np.array([0, 0, 40])
    upper_road = np.array([180, 90, 140])
    # lower_road = np.array([0, 0, 44])
    # upper_road = np.array([180, 57, 220])
    mask_road = cv2.inRange(hsv, lower_road, upper_road)

    # --- Step 2: Suppress green & sky ---
    lower_green = np.array([30, 60, 40])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_road[mask_green > 0] = 0  # remove green areas from road mask

    # --- Step 3: Morphological cleaning ---
    close_k = np.ones((close_kernel, close_kernel), np.uint8)
    open_k = np.ones((open_kernel, open_kernel), np.uint8)
    mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_CLOSE, close_k)
    mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_OPEN, open_k)

    # --- Step 4: Strengthen the road area ---
    mask_road = cv2.dilate(mask_road, np.ones((min_road_width, min_road_width), np.uint8))
    road_pixel_count = cv2.countNonZero(mask_road)
    # print("White pixel",road_pixel_count)
    # --- Step 5: Compose clean output ---
    new_img = np.full_like(img, (100, 202, 100))  # background green

    if add_grass:
        add_random_grass_patches(new_img, patch_size=150, patch_count=3)

    # Paint road gray
    new_img[mask_road > 0] = (102, 102, 102)

    # --- Step 6: Add car representation ---
    draw_gym_car_shape(new_img, center=(320, 410), scale=0.25)

    return new_img

def vae_out_to_cv2(recon):
    """
    recon: torch BCHW in [0,1]
    returns: numpy HWC uint8 BGR (for cv2.imshow)
    """
    recon_np = recon[0].permute(1, 2, 0).detach().cpu().numpy()  # HWC float
    recon_np = (recon_np * 255).clip(0, 255).astype(np.uint8)
    return recon_np  # still BGR ordering like your training code

# ======================================================
# LOSS
# ======================================================
def vae_loss(recon, target, mu, logvar, beta=0.001):
    recon_loss = F.mse_loss(recon, target)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
   
   
    return recon_loss #+ beta * kl_loss

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

def np_img_to_cv2(img):
    """
    img: numpy HWC float [0,1] or uint8
    """
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    return img[:, :, ::-1]  # RGB → BGR

def torch_img_to_cv2(img):
    """
    img: torch CHW float [0,1]
    """
    img = img.permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img[:, :, ::-1]  # RGB → BGR

# ======================================================
# TRAINING
# ======================================================
def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    # device = "cpu"
    # model_path = "path/to/saved_model.pth"

    for vae_model_path in PATCH_MODEL_PATHS:
        print(f"[RUN] {vae_model_path}")
        model_path = model_path("results", "CarRacing-v2_0_pomdp", "20_dec_gazebo", "models_best_8.pth")  #31_oct_gym  7_nov_Gazebo
        config = RacingCarConfig(capacity=1)
        RSSMModel, ObsEncoderModel, ActionModel = load_model(config, model_path, "cpu")

        model = VAE().to(device)
        model.load_state_dict(torch.load(vae_model_path, map_location=device,weights_only=True))
        model.eval()


        data = torch.load(TEST_DATA_PATH)
        images = data["srcimage"]  # numpy (84,84,3)
        prev_action = data["prev_action"]
        prev_rssmstate_ = data["prev_rssmstate"]
        model_state_ = data["model_state"]

        done = False
        mse1 = 0
        mse2 = 0
        mse1_ = []
        mse2_ = []

        nmse1_ =  []
        nmse2_ =  []

        file_saved_path = vae_model_path.parent
        for i in range(len(images)):

            img_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            muddy_ = add_mud_patch(img_rgb)
            muddy = muddy_.unsqueeze(0).to(device)
            with torch.no_grad():
                recon, mu, logvar = model(muddy)

            # ---- Convert for display ----
            orig_show  = np_img_to_cv2(img_rgb)
            muddy_show = torch_img_to_cv2(muddy_)
            recon_show = vae_out_to_cv2(recon)

            orig_show_t = patched_transform(orig_show)
            muddy_show_t = patched_transform(muddy_show)
            recon_show_t = patched_transform(recon_show)

            # ---- Show ----
            cv2.imshow("Original", orig_show)
            cv2.imshow("Muddy", muddy_show)
            cv2.imshow("Reconstruction", recon_show)

            key = cv2.waitKey(30)
            if key == 27:  # ESC to quit
                break

            recon_t_ = np.expand_dims(recon_show_t,axis=0)
            muddy_t_ = np.expand_dims(muddy_show_t,axis=0)
            orig_t = np.expand_dims(orig_show_t,axis=0)

            # img_rgb = torch.from_numpy(img_rgb).float().to(device)
            # recon_show = torch.from_numpy(recon_show).float().to(device)
            # mse1 = F.mse_loss(img_rgb, recon_show)
            # mse1_.append(mse1.item())
            # print(mse1.item())


            embed1 = ObsEncoderModel(torch.tensor(orig_t, dtype=torch.float32)
                                            .unsqueeze(0).to("cpu"))
            _, posterior_rssm_state1 = RSSMModel.rssm_observe(
                embed1, prev_action[i], not done, prev_rssmstate_[i]
            )
            model_state1 = RSSMModel.get_model_state(posterior_rssm_state1)


            embed2 = ObsEncoderModel(torch.tensor(muddy_t_, dtype=torch.float32)
                                            .unsqueeze(0).to("cpu"))
            _, posterior_rssm_state2 = RSSMModel.rssm_observe(
                embed2, prev_action[i], not done, prev_rssmstate_[i]
            )
            model_state2 = RSSMModel.get_model_state(posterior_rssm_state2)

            embed3 = ObsEncoderModel(torch.tensor(recon_t_, dtype=torch.float32)
                                            .unsqueeze(0).to("cpu"))
            _, posterior_rssm_state3 = RSSMModel.rssm_observe(
                embed3, prev_action[i], not done, prev_rssmstate_[i]
            )
            model_state3 = RSSMModel.get_model_state(posterior_rssm_state3)


            z_dim = 1023                # confirm once if needed
            z_original = model_state1[:, :z_dim] # torch.from_numpy(model_state_[i]).float().unsqueeze(0)[:, :z_dim]
            z_patched = model_state2[:, :z_dim]
            z_recon = model_state3[:, :z_dim]





            mse1 = F.mse_loss(z_original, z_patched)
            mse2 = F.mse_loss(z_original, z_recon)

            mse1_.append(mse1.item())
            mse2_.append(mse2.item())


            nmse1 = normalized_mse(z_patched,z_original).item()
            nmse2 = normalized_mse(z_recon,z_original).item()

            nmse1_.append(nmse1)
            nmse2_.append(nmse2)







        avg_mse_o_m = np.mean(mse1_)
        avg_mse_o_r = np.mean(mse2_)

        avg_nmse_o_m = np.mean(nmse1_)
        avg_nmse_o_r = np.mean(nmse2_)



        data_original_mud, cdf_original_mud = compute_cdf(mse1_)
        data_original_recon, cdf_original_recon = compute_cdf(mse2_)

        data_original_mud_norm, cdf_original_mud_norm = compute_cdf(nmse1_)
        data_original_recon_norm, cdf_original_recon_norm = compute_cdf(nmse2_)

        plt.figure(figsize=(8,5))
        plt.plot(data_original_mud, cdf_original_mud, label="test latent MSE (original vs mud)")
        plt.plot(data_original_recon, cdf_original_recon, label="test latent MSE (original vs recon)")
        plt.xlabel("Normalized MSE")
        plt.ylabel("CDF")
        plt.title("CDF of Normalized MSE")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(file_saved_path/"latent_cdf_mse.png", dpi=300)
        # plt.show()
        plt.close()


        plt.figure(figsize=(8,5))
        plt.plot(data_original_mud_norm, cdf_original_mud_norm, label="test latent NMSE (original vs mud)")
        plt.plot(data_original_recon_norm, cdf_original_recon_norm, label="test latent NMSE (original vs recon)")
        plt.xlabel("Normalized MSE")
        plt.ylabel("CDF")
        plt.title("CDF of Normalized MSE")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(file_saved_path/"latent_cdf_nmse.png", dpi=300)
        # plt.show()
        plt.close()


        print(np.sum(mse1_)/len(images))
        np.save("mse1_Z.npy", np.array(mse1_))
        np.save("mse2_Z_.npy", np.array(mse2_))

        print(f"Original and Mud latent error average: {np.mean(mse1_):.6f}")
        print(f"Original and Recon latent error average: {np.mean(mse2_):.6f}")

        print(f"Original and Mud norm latent error average: {np.mean(nmse1_):.6f}")
        print(f"Original and Recon norm latent error average: {np.mean(nmse2_):.6f}")

        plt.figure(figsize=(8,5))
        plt.plot(mse1_, label="Original and Mud latent error")
        plt.plot(mse2_, label="Original and Recon latent error")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(file_saved_path /"latent_mse.png", dpi=300)
        # plt.show()
        plt.close()


        plt.figure(figsize=(8,5))
        plt.plot(nmse1_, label="Original and Mud norm latent error")
        plt.plot(nmse2_, label="Original and Recon norm latent error")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(file_saved_path /"latent_nmse.png", dpi=300)
        # plt.show()
        plt.close()


        timing_path = file_saved_path /"latent_info.txt"
        with open(timing_path, "w") as f:
            f.write("===== LOSS AVERAGES =====\n")
            f.write(f"Avg test latent MSE Loss (Mud vs Original): {avg_mse_o_m:.6f}\n")
            f.write(f"Avg test latent MSE Loss (Recon vs Original): {avg_mse_o_r:.6f}\n\n")
        
            f.write("===== NORMALIZED MSE (NMSE) =====\n")
            f.write(f"Avg test latent NMSE Loss (Mud vs Original): {avg_nmse_o_m:.6f}\n")
            f.write(f"Avg test latent NMSE Loss (Recon vs Original): {avg_nmse_o_r:.6f}\n")




if __name__ == "__main__":
    test()
