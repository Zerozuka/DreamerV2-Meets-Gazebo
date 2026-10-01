import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
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
TEST_DATA_PATH = model_path("masked_data", "proposed_results_2.pt")
MODEL_PATH = model_path("masked_data", "trained_model", "patch_model2.pt")
BATCH_SIZE = 1
NUM_SAMPLES_TO_SHOW = 10

MUD_COLOR = (14, 34, 49)  # BGR

# ======================================================
# MUD PATCH (SAME AS TRAINING)
# ======================================================
def add_mud_patch(image):
    """
    image: torch.Tensor [3,84,84] or numpy [84,84,3]
    returns: torch.Tensor [3,84,84]
    """

    if isinstance(image, torch.Tensor):
        img = (image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    else:
        img = image.copy()
        if img.dtype != np.uint8:
            img = (img * 255).astype("uint8")

    h, w, _ = img.shape
    overlay = img.copy()

    num_patches = random.randint(5, 7)
    for _ in range(num_patches):
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)
        r  = random.randint(8, 13)
        cv2.circle(overlay, (cx, cy), r, MUD_COLOR, -1)

    alpha = 1.0  # same as training
    muddy = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    muddy = torch.from_numpy(muddy).permute(2, 0, 1).float() / 255.0
    return muddy

# ======================================================
# DATASET
# ======================================================
class TestDataset(Dataset):
    def __init__(self, pt_path):
        data = torch.load(pt_path)
        self.images = data["srcimage"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        clean = self.images[idx]

        if isinstance(clean, np.ndarray):
            clean = torch.from_numpy(clean).permute(2, 0, 1).float()
            if clean.max() > 1:
                clean = clean / 255.0

        muddy = add_mud_patch(clean)
        return muddy, clean

# ======================================================
# VAE MODEL (SAME AS TRAINING)
# ======================================================
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 84, 84)
            h = self.encoder(dummy)
            self.enc_shape = h.shape[1:]
            self.flat_dim = h.numel()

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
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
        recon = F.interpolate(
            recon, size=(84, 84),
            mode="bilinear",
            align_corners=False
        )
        return recon

# ======================================================
# TESTING / INFERENCE
# ======================================================
def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    dataset = TestDataset(TEST_DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = VAE().to(device)
    print(model)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    shown = 0

    with torch.no_grad():
        for muddy, clean in loader:
            muddy = muddy.to(device)
            print(muddy.shape)
            clean = clean.to(device)

            recon = model(muddy)

            # ---- Convert to numpy for visualization ----
            clean_np = clean[0].permute(1, 2, 0).cpu().numpy()
            muddy_np = muddy[0].permute(1, 2, 0).cpu().numpy()
            recon_np = recon[0].permute(1, 2, 0).cpu().numpy()

            clean_np = cv2.cvtColor(clean_np, cv2.COLOR_BGR2RGB)
            muddy_np = cv2.cvtColor(muddy_np, cv2.COLOR_BGR2RGB)
            recon_np = cv2.cvtColor(recon_np, cv2.COLOR_BGR2RGB)

            # ---- Plot ----
            plt.figure(figsize=(9, 3))

            plt.subplot(1, 3, 1)
            plt.imshow(clean_np)
            plt.title("original")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(muddy_np)
            plt.title("patched")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(recon_np)
            plt.title("Reconstructed")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

            shown += 1
            if shown >= NUM_SAMPLES_TO_SHOW:
                break

# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    test()
