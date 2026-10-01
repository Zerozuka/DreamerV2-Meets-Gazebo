#!/usr/bin/env python3
import os
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
# Sionna RT 2.x は import 時に mitsuba の variant を cuda_ad_mono_polarized に
# 決めるが、OptiX のライブラリ (libnvoptix.so.1) が無い環境では load_scene() が
# "Could not initialize OptiX!" で落ちる。sionna.rt より前にバックエンドを
# 選んでおく (OptiX が使えなければ LLVM にフォールバックする)。
import os as _sc_os
import sys as _sc_sys
from ament_index_python.packages import get_package_share_directory as _sc_share
_sc_sys.path.insert(0, _sc_os.path.join(_sc_share('gz_sionna'), 'src'))
import sionna_compat as _sionna_compat
_sionna_compat.select_backend()

import sionna.rt
import os
import mitsuba as mi
# import sionna_vispy
import matplotlib.pyplot as plt
import numpy as np
from sionna.rt import Scene, load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, RadioMapSolver, ITURadioMaterial, SceneObject #,cir_to_ofdm_channel
from sionna.phy.channel import subcarrier_frequencies , cir_to_ofdm_channel
from dreamerv2.models.actor import DiscreteActionModel
from dreamerv2.models.rssm import RSSM
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.pixel import ObsDecoder, ObsEncoder
from dreamerv2.training.config_ import RacingCarConfig
from gazebo_env import GazeboEnv
from gazebo_wrappers import ImageEnv, OneHotAction
from dreamerv2.training.config_ import RacingCarConfig

# --- リポジトリ内資産のパス解決 -------------------------------------------
# 元は開発者の home を指す絶対パス (/home/icon-group/...) がハードコードされて
# いたが、実体は gz_sionna パッケージに同梱されている。ament_index で share の
# 位置を引いて組み立てる。
from ament_index_python.packages import get_package_share_directory as _gz_share
# paths.py は gz_sionna/src に置いて全パッケージで共有している。
# gz_sionna はまだ Python モジュールを install していないので share のパスを通す。
import os as _p_os
import sys as _p_sys
from ament_index_python.packages import get_package_share_directory as _p_share
_p_sys.path.insert(0, _p_os.path.join(_p_share('gz_sionna'), 'src'))
from paths import model_path, output_path


def _gz_model(*parts):
    """gz_sionna の models/ 配下のパスを返す。"""
    return os.path.join(_gz_share("gz_sionna"), "models", *parts)
# --------------------------------------------------------------------------


no_preview = False



class Point:
    def __init__(self, x=0, y=0, z=0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

def main():
    
    robot1_pos = Point()
    robot1_orien = Point()

    scene = load_scene(_gz_model("with_materials", "untitled.xml")) 
    car_path = _gz_model("jetbot_real", "jet.obj")
    dataset_file = model_path("Proposed", "case_0", "proposed_results.pt")
    save_file = output_path("Proposed", "case_0")


    data = torch.load(dataset_file, map_location="cpu",weights_only=False)
    poses = data["poses"]
    # model_states = data["latent_state"]

    if hasattr(poses, "numpy"):
        poses_np = _sionna_compat.to_numpy(poses).astype(np.float32)
    else:
        poses_np = np.array(poses, dtype=np.float32)

    # if hasattr(model_states, "numpy"):
    #     model_states_np = model_states.numpy().astype(np.float32)
    # else:
    #     model_states_np = np.array(model_states, dtype=np.float32)


    num_cars = 1
    car_material = ITURadioMaterial("car-material",
                                "metal",
                                thickness=0.0001,
                                color=(0, 0, 1))

    cars = [SceneObject(fname=car_path ,   #sionna.rt.scene.low_poly_car, # Simple mesh of a car
                    name=f"car-{i}",
                    radio_material=car_material)
        for i in range(num_cars)]

    scene.edit(add=cars)
    
    i = 0
    my_cam = Camera(position=[40,30,60], look_at=[0,0,-10])
    scene.render(camera=my_cam, resolution=[650, 500], num_samples=512);
    my_cam2 = Camera(position=[0,0,100], look_at=[0,0,0])
    scene.render(camera=my_cam2, resolution=[650, 500], num_samples=512);
    my_cam3 = Camera(position=[-40,30,60], look_at=[0,0,10])
    scene.render(camera=my_cam3, resolution=[650, 500], num_samples=512);
    print("Cameras rendered.")

    scene.tx_array = PlanarArray(
        num_rows=2,
        num_cols=4,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )

    robot1_pos.x = float(poses_np[0][0])
    robot1_pos.y = float(poses_np[0][1])
    robot1_pos.z = float(0.5) #position.z

    tx = Transmitter("tx", position= [robot1_pos.x, robot1_pos.y, robot1_pos.z],display_radius=0.1)
   
    rx1 = Receiver("rx_1", position=[11.561232 ,12.377551 ,6.837],display_radius=0.1)
    rx2 = Receiver("rx_2", position=[-11.283523,-5.880425,  6.837],display_radius=0.1) 
    rx3 = Receiver("rx_3", position=[-9.654398, 11.918360, 6.837],display_radius=0.1)

    scene.add(tx)
    scene.add(rx1)
    scene.add(rx2)
    scene.add(rx3)
    # scene.add(rx4)
    # scene.add(rx5)

    scene.frequency = 2.14e9
    scene.synthetic_array = True
    subcarrier_spacing = 20e6 / 16  # 15e3
    fft_size = 16
    frequencies = subcarrier_frequencies(fft_size, subcarrier_spacing)

    channels = []

    # rm_solver = RadioMapSolver()
    # p_solver = PathSolver()
    path_solver = PathSolver()
    dataset = {"poses": [],"channels": []}
    USE_MONTE_CARLO = False  # GPU machine
    tx.look_at(rx1) 
    tx.look_at(rx2) 
    tx.look_at(rx3)
    # tx.look_at(rx4)
    # tx.look_at(rx5)
    for idx in range(poses_np.shape[0]):
        robot1_pos.x = float(poses_np[idx][0])
        robot1_pos.y = float(poses_np[idx][1])
        robot1_pos.z = float(0.12) #position.z

        robot1_orien.x = float(0)
        robot1_orien.y = float(0)
        robot1_orien.z = float(0)

        i = i + 1 
        img_path = save_file+"scene/img/scene_" + str(i)+".png"
        img_path2 = "scene/img2/scene_" + str(i)+".png"
        img_path3 = save_file+"scene/img3/scene_" + str(i)+".png"
        graph_path = "scene/graph/scene_" + str(i)+".png"
        
        # print("position updating..")
        # cars[0].position = mi.Point3f(robot1_pos.x, robot1_pos.y, robot1_pos.z)
        # cars[0].look_at(mi.Point3f(robot1_orien.x, robot1_orien.y, robot1_orien.z))
    
        scene.get("tx").position = mi.Point3f(robot1_pos.x, robot1_pos.y, robot1_pos.z)
        # scene.update()
        tx.look_at(rx1) 
        tx.look_at(rx2) 
        tx.look_at(rx3)
        # tx.look_at(rx4)
        # tx.look_at(rx5)
        

        # if USE_MONTE_CARLO == True:
        #     paths = scene.compute_paths(max_depth=5,num_samples=1e6)
            
        # else:
        paths = path_solver(scene,max_depth=5,samples_per_src=1000000)
           


        # Sionna 2.x では cir_to_ofdm_channel が torch テンソルとバッチ次元を
        # 要求する (PHY が PyTorch 化されたため)。out_type="numpy" のままだと
        #   AttributeError: 'numpy.ndarray' object has no attribute 'dim'
        # で落ちる。型合わせとバッチ次元の追加は sionna_compat に寄せている。
        a, tau = _sionna_compat.cir_for_ofdm(paths, normalize_delays=True)
        # t = tau.reshape(-1) / 1e-9          # ns
        # a_abs = np.abs(a).reshape(-1)

        print("Shape of a: ", a.shape)
        print("Shape of tau: ", tau.shape)


        # Sionna 1.x では cir_to_ofdm_channel が TensorFlow のテンソルを返すため
        # torch.from_numpy(....numpy()) で変換していた。2.x では PHY が
        # PyTorch 化され最初から torch テンソルが返るので往復は不要である。
        h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=False)
        h_time = torch.fft.fft(h_freq)
        channels.append(_sionna_compat.to_numpy(h_freq).squeeze())
        channel_np = _sionna_compat.to_numpy(h_time).squeeze()
        # channel_np = np.array(h_freq).squeeze()
        # print(channel_np.shape)

        # if(a.shape[5]>0):
        #     dataset["poses"].append(poses_np[idx])
        #     dataset["channel"].append(channel_np)

        #     print(f"[{i}/{poses_np.shape[0]}] Channel computed. Shape = {channel_np.shape}")
        
        # else:
        #     break


        

        # # # And plot the CIR
        # plt.figure()
        # plt.title("Channel impulse response")
        # plt.stem(t, a_abs)
        # plt.xlabel(r"$\tau$ [ns]")
        # plt.ylabel(r"$|a|$");
        # plt.savefig(graph_path)
        # plt.close()

       
        if not no_preview:
            if(a.shape[5]>0):
                
                scene.render_to_file(camera=my_cam,
                         filename=img_path,
                         resolution=[650,500],
                         paths =paths );
    
                # scene.render_to_file(camera=my_cam2,
                #          filename=img_path2,
                #          resolution=[650,500],
                #          paths =paths);
    
                scene.render_to_file(camera=my_cam3,
                         filename=img_path3,
                         resolution=[650,500],
                         paths =paths);

    # save_path = model_path("baseline", "channel_baseline_case_8.pt") 
    # torch.save(dataset, save_path)
    # print(f"Saved dataset with channels → {save_path}")

        
        

        


if __name__ == "__main__":
    main()
    