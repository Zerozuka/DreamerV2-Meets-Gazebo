#!/usr/bin/env python3

# ROS 1 の rospy を rclpy の上に再現する移植用の層に差し替えている。
# 本来はこのスクリプト自身が Node を持つべきだが、学習済み重みが無くて実行
# 検証できないため、意味を保つ層を挟んで呼び出し側を無改修にしている。
# 詳細は gz_sionna/src/ros1_compat.py の docstring を参照。
#
# gz_sionna はまだ Python モジュールを install していない (dreamerv2 / utils の
# 名前衝突を解消するまで packages=[] のため) ので share のパスを通す。
import os as _os
import sys as _sys
from ament_index_python.packages import get_package_share_directory as _share
_sys.path.insert(0, _os.path.join(_share('gz_sionna'), 'src'))
import ros1_compat as rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
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
import tf_transformations
import mitsuba as mi
from std_msgs.msg import Float32MultiArray,MultiArrayDimension
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist
from threading import Lock
import time
import matplotlib.pyplot as plt
import numpy as np
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,PathSolver, RadioMapSolver, subcarrier_frequencies, ITURadioMaterial, SceneObject
from sionna.phy.channel import subcarrier_frequencies , cir_to_ofdm_channel

# --- リポジトリ内資産のパス解決 -------------------------------------------
# 元は開発者の home を指す絶対パス (/home/icon-group/...) がハードコードされて
# いたが、実体は gz_sionna パッケージに同梱されている。ament_index で share の
# 位置を引いて組み立てる。
from ament_index_python.packages import get_package_share_directory as _gz_share
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__),
    "..", "..", "control_jepa", "test"))
# paths.py は gz_sionna/src に置いて全パッケージで共有している。
# gz_sionna はまだ Python モジュールを install していないので share のパスを通す。
import os as _p_os
import sys as _p_sys
from ament_index_python.packages import get_package_share_directory as _p_share
_p_sys.path.insert(0, _p_os.path.join(_p_share('gz_sionna'), 'src'))
from paths import output_path


def _gz_model(*parts):
    """gz_sionna の models/ 配下のパスを返す。"""
    return os.path.join(_gz_share("gz_sionna"), "models", *parts)
# --------------------------------------------------------------------------



robot1_pos = Point(x=0.0, y=0.0, z=0.0)
robot1_orien = Point(x=0.0, y=0.0, z=0.0)

robot1_vel = Point(x=0.0, y=0.0, z=0.0)

no_preview = False
render_flag = 0
render_lock = Lock()
def odom_callback1(msg):
    global robot1_pos, robot1_orien
    position = msg.pose.pose.position
    robot1_pos.x = position.x
    robot1_pos.y = position.y
    robot1_pos.z = 0.12#position.z

    q = msg.pose.pose.orientation
    quaternion = (q.x, q.y, q.z, q.w)

    roll, pitch, yaw = tf_transformations.euler_from_quaternion(quaternion)

    robot1_orien.x = roll
    robot1_orien.y = pitch
    robot1_orien.z = yaw

def cmd_vel_callback(msg):
    global robot1_vel
    robot1_vel.x = msg.linear.x
    robot1_vel.y = msg.linear.y
    robot1_vel.z = msg.linear.z


def render_callback(msg):
    global render_flag
    render_flag = msg.data   # 1 = render, 0 = do nothing

def main():
    global robot1_pos, render_flag,robot1_vel
    rospy.init_node("wireless_channel")


    rospy.Subscriber("/odom", Odometry, odom_callback1)
    rospy.Subscriber("/render_trigger", Int32, render_callback)
    rospy.Subscriber("/cmd_vel", Twist, cmd_vel_callback)   

    pub = rospy.Publisher("/channels", Float32MultiArray, queue_size=10)
    done_pub = rospy.Publisher("/render_done", Int32, queue_size=10)
    rospy.loginfo("Subscribed to /robot_position")

    scene = load_scene(_gz_model("with_materials", "untitled.xml")) 
    car_path = _gz_model("jetbot_real", "jet.obj")
    cube_path = _gz_model("jepa_objects", "cube.obj")
    ball_path = _gz_model("jepa_objects", "ball.obj")
    cylinder_path = _gz_model("jepa_objects", "cylinder.obj")



    output_dir = output_path("Proposed", "case_0")
    os.makedirs(output_dir, exist_ok=True)

    cube_material = ITURadioMaterial(
    "cube-material",
    "concrete",
    thickness=1,
    color=(1, 1, 0)
    )

    ball_material = ITURadioMaterial(
        "ball-material",
        "metal",
        thickness=1,
        color=(0, 1, 1)
    )

    cylinder_material = ITURadioMaterial(
        "cylinder-material",
        "wood",
        thickness=1,
        color=(1, 0, 1)
    )


    cube_positions = [
        (7.809810,  8.423260, 0.5),
        (2.814624,  0.555732, 0.5),
        (10.178213, 7.527594, 0.5),
        (4.503686, -8.171050, 0.5),
        (-10.628513, 3.182821, 0.5),
        (-1.856461, 8.704827, 0.5),
        (-7.197225, -6.432545, 0.5),
        (-11.255310, -4.816126, 0.5),
    ]

    ball_positions = [
        (1.683725,  8.405119, 0.5),
        (8.686228, -2.047810, 0.5),
        (9.244602, 10.747196, 0.5),
        (2.328079, -8.332348, 0.5),
        (-10.061080, -8.313878, 0.5),
        (7.762015,  3.934394, 0.5),
        (6.192001, -7.197225, 0.5),
        (-10.707787, 8.113759, 0.5),
    ]

    cylinder_positions = [
        (2.323899,  2.778099, 0.5),
        (2.411660, -2.047805, 0.5),
        (9.274621,  1.227368, 0.5),
        (-11.150761, -2.211683, 0.5),
        (-11.712104,  9.665686, 0.5),
        (10.670957, -5.428221, 0.5),
        (-8.323398, -8.323398, 0.5),
        (-10.486228, 11.339570, 0.5),
    ]


    num_ball = 8
    balls = [SceneObject(fname=ball_path , 
                    name=f"ball-{i}",
                    radio_material=ball_material)
        for i in range(num_ball)]
    
    num_cubes = 8
    cubes = [SceneObject(fname=cube_path , 
                    name=f"cube-{i}",
                    radio_material=cube_material)
        for i in range(num_cubes)]


    num_cylinder = 8
    cylinders = [SceneObject(fname=cylinder_path , 
                    name=f"cylinder-{i}",
                    radio_material=cylinder_material)
        for i in range(num_cylinder)]
    


    # scene.edit(add=balls)
    # scene.edit(add=cubes)
    # scene.edit(add=cylinders)

    # for cube, pos in zip(cubes, cube_positions):
    #     cube.position = mi.Point3f(*pos)

    # for ball, pos in zip(balls, ball_positions):
    #     ball.position = mi.Point3f(*pos)

    # for cyl, pos in zip(cylinders, cylinder_positions):
    #     cyl.position = mi.Point3f(*pos)


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
 

    tx = Transmitter("tx", position= [robot1_pos.x, robot1_pos.y, robot1_pos.z],display_radius=0.1,velocity=[robot1_vel.x,robot1_vel.y,robot1_vel.z])

   
    rx1 = Receiver("rx_1", position=[11.561232 ,12.377551 ,6.837],display_radius=0.1)
    rx2 = Receiver("rx_2", position=[-11.283523,-5.880425,  6.837],display_radius=0.1) 
    rx3 = Receiver("rx_3", position=[-9.654398, 11.918360, 6.837],display_radius=0.1)

    # # rx4 = Receiver("rx_4", position=[10.028879,-5.885381, 6.837],display_radius=0.1)
    # # rx5 = Receiver("rx_5", position=[-1.3,2.5,  6.837],display_radius=0.1)

    # rx1 = Receiver("rx_1", position=[26.627333, 6.138611 , 6.837],display_radius=0.1)  
    # rx2 = Receiver("rx_2", position=[-32.122665, 11.765728, 6.837 ],display_radius=0.1) 
    # rx3 = Receiver("rx_3", position=[-32.122665, -7.217361,  6.837 ],display_radius=0.1)  
     
    # rx4 = Receiver("rx_4", position=[8.238777, -5.601792, 3.077197 ],display_radius=0.1) 
    # rx5 = Receiver("rx_5", position=[-4.008763 ,0.092765 ,3.274808 ],display_radius=0.1) 

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
    
    USE_MONTE_CARLO = False  # GPU machine
    tx.look_at(rx1) 
    tx.look_at(rx2) 
    tx.look_at(rx3)
    # tx.look_at(rx4)
    # tx.look_at(rx5)
    EPISODE_SEED = int(time.time())
    while not rospy.is_shutdown():

        # while(render_flag != 1 and i!=0):
        #     fjgj=0
        
        img_path = output_dir+"scene/img/scene_" + str(i)+".png"
        img_path2 = output_dir+"scene/img2/scene_" + str(i)+".png"
        img_path3 = output_dir+"scene/img3/scene_" + str(i)+".png"
        graph_path = output_dir+"scene/graph/scene_" + str(i)+".png"
        
        # print("position updating..")
        # cars[0].position = mi.Point3f(robot1_pos.x, robot1_pos.y, robot1_pos.z)
        # cars[0].look_at(mi.Point3f(robot1_orien.x, robot1_orien.y, robot1_orien.z))
    
        scene.get("tx").position = mi.Point3f(robot1_pos.x, robot1_pos.y, robot1_pos.z)

        tx_velocity = [robot1_vel.x, robot1_vel.y, robot1_vel.z]
        scene.get("tx").velocity = tx_velocity

        # scene.update()
        tx.look_at(rx1) 
        tx.look_at(rx2) 
        tx.look_at(rx3)
        # tx.look_at(rx4)
        # tx.look_at(rx5)
        

        # if USE_MONTE_CARLO == True:
        #     paths = scene.compute_paths(max_depth=5,num_samples=1e6)
            
        # else:
        paths = path_solver(scene,max_depth=5,samples_per_src=1000000) #,seed=EPISODE_SEED
           


        # Sionna 2.x では cir_to_ofdm_channel が torch テンソルを要求する
        # (PHY が PyTorch 化されたため)。out_type="numpy" のままだと
        #   AttributeError: 'numpy.ndarray' object has no attribute 'dim'
        # で落ちる。バッチ次元は下の reshape で足しているのでここでは型だけ。
        a, tau = _sionna_compat.cir_for_ofdm(paths, normalize_delays=True)
        # t = tau.reshape(-1) / 1e-9          # ns
        # a_abs = np.abs(a).reshape(-1)

        # print("Shape of a: ", a.shape)
        # print("Shape of tau: ", tau.shape)


        h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=False)
        channels.append(h_freq.numpy().squeeze())
        channel_np = h_freq.numpy().squeeze()
        # print(channel_np.shape , channel_np.dtype)
        
        # print(channel_np.shape)
        # print(channel_np[0][0])
        stacked = np.stack([channel_np.real, channel_np.imag], axis=0)  
        # print(stacked.shape)
        
        msg = Float32MultiArray()
        msg.data = stacked.astype(np.float32).flatten().tolist()

        dim0 = MultiArrayDimension()
        dim0.label = "channels"
        dim0.size = 2
        dim0.stride = 3 * 8 * 16

        dim1 = MultiArrayDimension()
        dim1.label = "d1"
        dim1.size = 3
        dim1.stride = 8 * 16

        dim2 = MultiArrayDimension()
        dim2.label = "d2"
        dim2.size = 8
        dim2.stride = 16

        dim3 = MultiArrayDimension()
        dim3.label = "d3"
        dim3.size = 16
        dim3.stride = 1

        msg.layout.dim = [dim0, dim1, dim2, dim3]

        pub.publish(msg)


        
        # print(paths)
        # # # And plot the CIR
        # plt.figure()
        # plt.title("Channel impulse response")
        # plt.stem(t, a_abs)
        # plt.xlabel(r"$\tau$ [ns]")
        # plt.ylabel(r"$|a|$");
        # plt.savefig(graph_path)
        # plt.close()

       
        if not no_preview :
            
            # if(a.shape[5]>0):
            with render_lock:
                if (render_flag == 1):
                    
                    # i = i + 1 
                    # scene.render_to_file(camera=my_cam,
                    #             filename=img_path,
                    #             resolution=[650,500], 
                    #             paths =paths); 

                    # # scene.render_to_file(camera=my_cam2,
                    # #          filename=img_path2,
                    # #          resolution=[650,500],
                    # #          paths =paths);

                    # scene.render_to_file(camera=my_cam3,
                    #             filename=img_path3,
                    #             resolution=[650,500],
                    #             paths =paths); 
                    # mi.Thread.wait_for_tasks()
                    done_pub.publish(1)



        # done_pub.publish(1)
        render_flag = 0

       
        

        


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass