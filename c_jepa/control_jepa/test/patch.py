# import cv2
# import numpy as np
# import random


# # MUD_COLOR = (0, 0, 0)  # fixed mud color (BGR)
# MUD_COLOR = (14,34,49)

# def add_mud_patch(image, num_patches=6):
#     h, w, _ = image.shape
#     overlay = image.copy()
#     num_patches = random.randint(5, 7)
#     for _ in range(num_patches):
#         cx = random.randint(0, w)
#         cy = random.randint(0, h)
#         radius = random.randint(8, 13)

#         cv2.circle(overlay, (cx, cy), radius, MUD_COLOR, -1)

#     alpha = 1  # wet mud transparency
#     return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

# # Load image or camera frame
# img = cv2.imread(output_path("img.png"))   # OR frame from camera
# img= cv2.resize(img,(84,84))
# muddy = add_mud_patch(img)

# # Show only
# cv2.imshow("Muddy Camera View", muddy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import torch
import numpy as np
from paths import model_path, output_path
data_path = model_path("masked_data", "proposed_results.pt")

data = torch.load(data_path)
img = data["srcimage"][0]

print(type(img))
print(img.shape)