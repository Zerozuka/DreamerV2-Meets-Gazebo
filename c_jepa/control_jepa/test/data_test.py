import torch
import numpy as np
# paths.py は gz_sionna/src に置いて全パッケージで共有している。
# gz_sionna はまだ Python モジュールを install していないので share のパスを通す。
import os as _p_os
import sys as _p_sys
from ament_index_python.packages import get_package_share_directory as _p_share
_p_sys.path.insert(0, _p_os.path.join(_p_share('gz_sionna'), 'src'))
from paths import model_path

pt_path = model_path("Proposed", "case_2", "proposed_results.pt")

print("Loading:", pt_path)
data = torch.load(pt_path)

print("\n--- Dataset keys ---")
for k in data.keys():
    print(" ", k)

print("\n--- Counts ---")
for k in data:
    print(f"{k}: {len(data[k])}")

print("\n--- First entries ---")
print("step:", data["step"][0])
print("pose:", data["poses"][0])
print("reward:", data["reward"][0])
# prsint("channels:", data["channels"][0])
ss = 0
for i in range( len(data["uplink_com_status"])):
    if(data["uplink_com_status"][i]==0):
        # print(data["reward"][i])
        ss += 1 #data["reward"][i]


print("\n--- Channels info ---")
ch = data["channels"][0]
print("Type:", type(ch))
print("Shape:", ch.shape)
print("dtype:", ch.dtype)
print(ss)
# print("min:", np.min(ch))
# print("max:", np.max(ch))
