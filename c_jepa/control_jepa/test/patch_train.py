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
import time
from dreamerv2.training.config_ import RacingCarConfig
from tqdm.auto import tqdm
import time
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
case_id = 9
output_dir = model_path("masked_data", "trained_model")
DATA_PATH = model_path("masked_data", "proposed_results.pt")
BATCH_SIZE = 64
EPOCHS = 200
LR = 1e-3
LATENT_DIM = 128
VAL_RATIO = 0.5


MUD_COLOR = (14, 34, 49)  # BGR (OpenCV)
episode_id = 0
seed = 42


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
def add_mud_patch(image):
    
    # ---- to numpy HWC uint8 ----
    if isinstance(image, torch.Tensor):
        img = (image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    else:
        img = image.copy()
        if img.dtype != np.uint8:
            img = (img * 255).astype("uint8")

    h, w, _ = img.shape
    overlay = img.copy()

    num_patches = random.randint(7, 10)
    for _ in range(num_patches):
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)
        r  = random.randint(15, 20)
        cv2.circle(overlay, (cx, cy), r, MUD_COLOR, -1)

    alpha = 1.0  # hard mud (can change to 0.6 if needed)
    muddy = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # ---- back to torch CHW float ----
    muddy = torch.from_numpy(muddy).permute(2, 0, 1).float() / 255.0
    return muddy

# def add_mud_patch(image):
#     """
#     image: torch.Tensor [3,84,84] or numpy [84,84,3]
#     returns: torch.Tensor [3,84,84]
#     """

#     # ---- to numpy HWC uint8 ----
#     rng = random.Random(seed)
#     if isinstance(image, torch.Tensor):
#         img = (image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
#     else:
#         img = image.copy()
#         if img.dtype != np.uint8:
#             img = (img * 255).astype("uint8")

#     h, w, _ = img.shape
#     overlay = img.copy()

#     num_patches = rng.randint(7, 10)
#     for _ in range(num_patches):
#         cx = rng.randint(0, w - 1)
#         cy = rng.randint(0, h - 1)
#         r  = rng.randint(15, 20)
#         cv2.circle(overlay, (cx, cy), r, MUD_COLOR, -1)

#     alpha = 1.0  # hard mud (can change to 0.6 if needed)
#     muddy = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

#     # ---- back to torch CHW float ----
#     muddy = torch.from_numpy(muddy).permute(2, 0, 1).float() / 255.0
#     return muddy


# ======================================================
# DATASET
# ======================================================
# class MudDenoiseDataset(Dataset):
#     def __init__(self, pt_path):
#         data = torch.load(pt_path)
#         self.images = data["srcimage"]  # numpy (84,84,3)
#         # self.prev_action = data["prev_action"]
#         # self.prev_rssmstate_ = data["prev_rssmstate"]
#         # self.model_state_ = data["model_state"]

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         clean = self.images[idx]
#         # prev_act = self.prev_action[idx]
#         # prev_rssm = self.prev_rssmstate_[idx]
#         # model_state = self.model_state_[idx]
#         # numpy -> torch
#         if isinstance(clean, np.ndarray):
#             clean = torch.from_numpy(clean).permute(2, 0, 1).float()
#             if clean.max() > 1:
#                 clean = clean / 255.0
#         # if random.random() < 0.9:
#         #     clean = geometric_rotate_only(clean)

#             # img_vis = clean.detach().cpu().permute(1,2,0).numpy()
#             # img_vis = (img_vis * 255).astype(np.uint8)
#             # img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

#             # cv2.imshow("img_", img_vis)
#             # cv2.waitKey(1)
#             # time.sleep(0.5)

#         muddy = add_mud_patch(clean)
#         return muddy, clean #, prev_act,prev_rssm,model_state


class MudDenoiseDataset(Dataset):
    def __init__(self, pt_path, rot_deg=15):
        data = torch.load(pt_path)
        self.images = data["srcimage"]   # numpy (84,84,3)
        self.rot_deg = rot_deg
        # self.base_seed = base_seed

    def __len__(self):
        return 3 * len(self.images)

    def __getitem__(self, idx):
        base_idx = idx // 3
        mode = idx % 3  # 0=orig, 1=left, 2=right

        clean = self.images[base_idx]
        if isinstance(clean, np.ndarray):
            clean = torch.from_numpy(clean).permute(2, 0, 1).float()
            if clean.max() > 1:
                clean = clean / 255.0

        # rotate clean
        if mode == 1:
            clean = rotate_tensor_img(clean, -self.rot_deg)  # left
        elif mode == 2:
            clean = rotate_tensor_img(clean, +self.rot_deg)  # right

        # make mud (seeded per (base_idx, mode) so it’s stable)
        # mud_seed = self.base_seed + base_idx * 10 + mode
        muddy = add_mud_patch(clean)

        return muddy, clean

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

        # recommended resize to 84×84
        recon = F.interpolate(
            recon, size=(84, 84),
            mode="bilinear",
            align_corners=False
        )

        return recon, mu, logvar





# def patched_transform(img):

#         img = cv2.resize(img,(640,480))
#         img =  transform_image_cv2(img)
    
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (84, 84))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img = img.astype(np.float32) / 255.0
#         return img

def patched_transform(img):
    
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
    img = transform_image_cv2(img)   # expects BGR HWC

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (84, 84))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0
    return img



def add_random_grass_patches(img, patch_size=40, patch_count=2):
        
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


def rotate_tensor_img(img_chw: torch.Tensor, angle_deg: float):
    
    img = (img_chw.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # HWC uint8
    h, w, _ = img.shape

    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
    rot = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    rot = torch.from_numpy(rot).permute(2, 0, 1).float() / 255.0
    return rot


# ======================================================
# LOSS
# ======================================================

def compute_cdf(data):
    data = np.asarray(data).flatten()   # ensure 1D
    data = np.sort(data)
    cdf = np.arange(1, len(data) + 1) / len(data)
    return data, cdf


def vae_loss(recon, target, mu, logvar, beta=0.001):
    recon_loss = F.mse_loss(recon, target)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
   
   
    return recon_loss #+ beta * kl_loss

def normalized_mse(pred, target, eps=1e-8):
   
    num = torch.sum((pred - target) ** 2)
    denom = torch.sum(target ** 2) + eps
    return num / denom

# ======================================================
# TRAINING
# ======================================================
def train():
    start_time = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    # device = "cpu"
    # model_path = "path/to/saved_model.pth"
    model_path = model_path("results", "CarRacing-v2_0_pomdp", "20_dec_gazebo", "models_best_8.pth")  #31_oct_gym  7_nov_Gazebo
    config = RacingCarConfig(capacity=1)
    RSSMModel, ObsEncoderModel, ActionModel = load_model(config, model_path, "cpu")


    full_dataset = MudDenoiseDataset(DATA_PATH)

    val_size = int(len(full_dataset) * VAL_RATIO)
    train_size = len(full_dataset) - val_size

    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    train_len = len(train_ds)
    val_len   = len(val_ds)


    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    out_dir = output_dir + str(EPOCHS)+"/" + str(train_len)+"/" +str(case_id)+"/"
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)

    model = VAE(LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = [], []
    mudtrain_loss, mudval_loss = [],[]
    done = False
    mse1_train = []
    mse2_train = []

    mse1_val = []
    mse2_val = []

    nmse_recon_train = []
    nmse_mud_train   = []

    nmse_recon_val = []
    nmse_mud_val   = []

    for epoch in range(EPOCHS):
        # -------- TRAIN --------
        model.train()
        train_loss = 0.0

        mud_tloss = 0
        clean_img_norm = []
        mse1 =0
        mse2 =0
        nmse_r = 0.0
        nmse_m = 0.0

        for muddy, clean in train_loader:
           
            muddy = muddy.to(device)
            clean = clean.to(device)


            
            recon, mu, logvar = model(muddy)
            loss = vae_loss(recon, clean, mu, logvar)
            loss2 = vae_loss(muddy, clean, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            mud_tloss += loss2.item()

            nmse_r += normalized_mse(recon, clean).item()
            nmse_m += normalized_mse(muddy, clean).item()


        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        mse1 /= len(train_loader)
        mse2 /= len(train_loader)

        mse1_train.append(mse1)
        mse2_train.append(mse2)

        mud_tloss /= len(train_loader)
        mudtrain_loss.append(mud_tloss)

        nmse_r /= len(train_loader)
        nmse_m /= len(train_loader)

        nmse_recon_train.append(nmse_r)
        nmse_mud_train.append(nmse_m)


        # -------- VALIDATION --------
        model.eval()
        val_loss = 0.0
        mud_vloss = 0.0

        mse3 = 0
        mse4 = 0
        nmse_r = 0.0
        nmse_m = 0.0

        # with torch.no_grad():
        #     for muddy, clean in val_loader:

                

        #         muddy = muddy.to(device)
        #         clean = clean.to(device)
                
        #         muddy = muddy.to(device)
        #         clean = clean.to(device)

                
                

        #         recon, mu, logvar = model(muddy)
        #         loss = vae_loss(recon, clean, mu, logvar)
        #         loss2 = vae_loss(muddy, clean, mu, logvar)

        #         nmse_r += normalized_mse(recon, clean).item()
        #         nmse_m += normalized_mse(muddy, clean).item()

        #         val_loss += loss.item()
        #         mud_vloss += loss2.item()

        # val_loss /= len(val_loader)
        # val_losses.append(val_loss)

        # mud_vloss /= len(val_loader)
        # mudval_loss.append(mud_vloss)

        # mse3 /= len(val_loader)
        # mse4 /= len(val_loader)

        # mse1_val.append(mse3)
        # mse2_val.append(mse4)

        # nmse_r /= len(val_loader)
        # nmse_m /= len(val_loader)

        # nmse_recon_val.append(nmse_r)
        # nmse_mud_val.append(nmse_m)


        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f}"
        )



    avg_train_loss = np.mean(train_losses)
    avg_val_loss   = np.mean(val_losses)

    avg_mud_train_loss = np.mean(mudtrain_loss)
    avg_mud_val_loss   = np.mean(mudval_loss)


    avg_nmse_recon_train = np.mean(nmse_recon_train)
    avg_nmse_mud_train   = np.mean(nmse_mud_train)

    avg_nmse_recon_val = np.mean(nmse_recon_val)
    avg_nmse_mud_val   = np.mean(nmse_mud_val)

        

    # Save model & losses
    model_saved_path = out_dir +"/patch_model.pt"
    torch.save(model.state_dict(),model_saved_path)
    np.save(out_dir+"train_losses.npy", np.array(train_losses))
    np.save(out_dir+"val_losses.npy", np.array(val_losses))

    np.save(out_dir+"clean_mud_train_losses.npy", np.array(mudtrain_loss))
    np.save(out_dir+"clean_mud_val_losses.npy", np.array(mudval_loss))

    np.save(out_dir+"mse1_val.npy", np.array(mse1_val))
    np.save(out_dir+"mse2_val.npy", np.array(mse2_val))

    np.save(out_dir+"mse1_train.npy", np.array(mse1_train))
    np.save(out_dir+"mse2_train.npy", np.array(mse2_train))


    # -------- PLOT CURVES --------
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.plot(mudtrain_loss, label="original img and mud img Loss train")
    plt.plot(mudval_loss, label="original img and mud img Loss val ")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir + "mseloss_curves.png", dpi=300)
    # plt.show()

    # plt.figure(figsize=(8,5))
    # plt.plot(mudtrain_loss, label="original img and mud img Loss train")
    # plt.plot(mudval_loss, label="original img and mud img Loss val ")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("VAE Training & Validation Loss")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    print("\n===== FINAL NORMALIZED MSE =====")
    print(f"Train NMSE (Recon vs Clean): {np.mean(nmse_recon_train):.6f}")
    print(f"Train NMSE (Mud   vs Clean): {np.mean(nmse_mud_train):.6f}")
    print(f"Val   NMSE (Recon vs Clean): {np.mean(nmse_recon_val):.6f}")
    print(f"Val   NMSE (Mud   vs Clean): {np.mean(nmse_mud_val):.6f}")


    plt.figure(figsize=(8,5))
    plt.plot(nmse_recon_train, label="Train NMSE (Recon / Clean)")
    plt.plot(nmse_recon_val,   label="Val NMSE (Recon / Clean)")
    plt.plot(nmse_mud_train,   label="Train NMSE (Mud / Clean)", linestyle="--")
    plt.plot(nmse_mud_val,     label="Val NMSE (Mud / Clean)", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized MSE")
    plt.title("Normalized MSE (||X̂ − X||² / ||X||²)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir + "nmse_curves.png", dpi=300)
    # plt.show()

    # ===============================
# CDF FIGURE — MSE
# ===============================

    data_tr, cdf_tr = compute_cdf(train_losses)
    data_val, cdf_val = compute_cdf(val_losses)

    data_mud_tr, cdf_mud_tr = compute_cdf(mudtrain_loss)
    data_mud_val, cdf_mud_val = compute_cdf(mudval_loss)

    plt.figure(figsize=(8,5))
    plt.plot(data_tr, cdf_tr, label="Train MSE (Recon vs Clean)")
    plt.plot(data_val, cdf_val, label="Val MSE (Recon vs Clean)")
    plt.plot(data_mud_tr, cdf_mud_tr, "--", label="Train MSE (Mud vs Clean)")
    plt.plot(data_mud_val, cdf_mud_val, "--", label="Val MSE (Mud vs Clean)")

    plt.xlabel("MSE")
    plt.ylabel("CDF")
    plt.title("CDF of MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir + "cdf_mse.png", dpi=300)
    # plt.show()


    # ===============================
    # CDF FIGURE — NMSE
    # ===============================

    data_tr, cdf_tr = compute_cdf(nmse_recon_train)
    data_val, cdf_val = compute_cdf(nmse_recon_val)

    data_mud_tr, cdf_mud_tr = compute_cdf(nmse_mud_train)
    data_mud_val, cdf_mud_val = compute_cdf(nmse_mud_val)

    plt.figure(figsize=(8,5))
    plt.plot(data_tr, cdf_tr, label="Train NMSE (Recon vs Clean)")
    plt.plot(data_val, cdf_val, label="Val NMSE (Recon vs Clean)")
    plt.plot(data_mud_tr, cdf_mud_tr, "--", label="Train NMSE (Mud vs Clean)")
    plt.plot(data_mud_val, cdf_mud_val, "--", label="Val NMSE (Mud vs Clean)")

    plt.xlabel("Normalized MSE")
    plt.ylabel("CDF")
    plt.title("CDF of Normalized MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir + "cdf_nmse.png", dpi=300)
    # plt.show()


    timing_path = out_dir + "timing_info.txt"
    os.makedirs(os.path.dirname(timing_path), exist_ok=True)
    end_time = time.time()
    total_training_time = end_time - start_time

    with open(timing_path, "w") as f:
        f.write("===== TRAINING METADATA =====\n")
        f.write(f"Total training time (seconds): {total_training_time:.2f}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Learning rate: {LR}\n\n")

        f.write("===== DATASET INFO =====\n")
        f.write(f"Train dataset length: {train_len}\n")
        f.write(f"Validation dataset length: {val_len}\n\n")

        f.write("===== LOSS AVERAGES =====\n")
        f.write(f"Avg Train MSE Loss (Recon vs Original): {avg_train_loss:.6f}\n")
        f.write(f"Avg Val MSE Loss (Recon vs Original): {avg_val_loss:.6f}\n\n")
        f.write(f"Avg Train MSE Loss (Mud vs Original): {avg_mud_train_loss:.6f}\n")
        f.write(f"Avg Val   MSE Loss (Mud vs Original): {avg_mud_val_loss:.6f}\n\n")


        f.write("===== NORMALIZED MSE (NMSE) =====\n")
        f.write(f"Train NMSE (Recon vs Original): {avg_nmse_recon_train:.6f}\n")
        f.write(f"Train NMSE (Mud   vs Original): {avg_nmse_mud_train:.6f}\n")
        f.write(f"Val   NMSE (Recon vs Original): {avg_nmse_recon_val:.6f}\n")
        f.write(f"Val   NMSE (Mud   vs Original): {avg_nmse_mud_val:.6f}\n")


    
# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    train()


