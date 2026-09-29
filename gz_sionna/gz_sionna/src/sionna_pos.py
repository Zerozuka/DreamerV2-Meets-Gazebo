#!/usr/bin/env python3.10

import rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
import sionna.rt
import os
import tf_transformations
import mitsuba as mi
import sionna_vispy


import matplotlib.pyplot as plt
import numpy as np
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, RadioMapSolver, subcarrier_frequencies, ITURadioMaterial, SceneObject

robot1_pos = Point()
robot2_pos = Point(0,0,0)

robot1_orien = Point()
robot2_orien = Point()

no_preview = False


def position_callback1(msg):
    global robot1_pos
    robot1_pos.x = round(msg.x,2)
    robot1_pos.y = round(msg.y,2)
    robot1_pos.z = round(msg.z,2)


def position_callback2(msg):
    global robot2_pos
    robot2_pos.x = round(msg.x,2)
    robot2_pos.y = round(msg.y,2)
    robot2_pos.z = round(msg.z,2)


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


def odom_callback2(msg):
    global robot2_pos, robot2_orien
    position = msg.pose.pose.position
    robot2_pos.x = position.x
    robot2_pos.y = position.y
    robot2_pos.z = 0.12 #position.z

    q = msg.pose.pose.orientation
    quaternion = (q.x, q.y, q.z, q.w)

    roll, pitch, yaw = tf_transformations.euler_from_quaternion(quaternion)

    robot2_orien.x = roll
    robot2_orien.y = pitch
    robot2_orien.z = yaw


def main():
    global robot1_pos,robot2_pos
    rospy.init_node("position_subscriber")

    # rospy.Subscriber("/jb_0/pos", Point, position_callback1)
    # rospy.Subscriber("/jb_1/pos", Point, position_callback2)

    rospy.Subscriber("/odom", Odometry, odom_callback1)
    # rospy.Subscriber("/tb3_1/odom", Odometry, odom_callback2)

    rospy.loginfo("Subscribed to /robot_position")

    scene = load_scene("/home/icon-group/catkin_ws/src/gz_sionna/gz_sionna/models/with_materials/untitled.xml") 
    car_path = "/home/icon-group/catkin_ws/src/gz_sionna/gz_sionna/models/jetbot_real/jet.obj"

    # if not no_preview:
    #         scene.preview();
    

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
    
    # cars[i].look_at(mi.Point3f(look_at_points.x[i], look_at_points.y[i], look_at_points.z[i]))

    
    i = 0
    my_cam = Camera(position=[20,10,50], look_at=[0,0,-10])
    scene.render(camera=my_cam, resolution=[650, 500], num_samples=512);

    my_cam2 = Camera(position=[4,6,2.5], look_at=[0,0,0])
    scene.render(camera=my_cam2, resolution=[650, 500], num_samples=512);

    scene.add(Transmitter("tx", position= [0,0,5], orientation=[0,0,0],display_radius=0.1))
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="tr38901", polarization="V")
    scene.rx_array = scene.tx_array

    rx = Receiver("rx", position=[robot2_pos.x, robot2_pos.y, robot2_pos.z], display_radius=0.1)
    scene.add(rx)


    
    
    # scene.render(camera=my_cam, radio_map=rm,
    #              num_samples=512, rm_show_color_bar=True,
    #              rm_vmax=-40, rm_vmin=-150)

    # with sionna_vispy.patch():
    #     canvas = scene.preview()
    #     canvas.show()
         
    rm_solver = RadioMapSolver()
    p_solver = PathSolver()
    while not rospy.is_shutdown():
        # global robot1_pos,robot2_pos

        i = i + 1 
        img_path = "img/scene_" + str(i)+".png"
        img_path2 = "img2/scene_" + str(i)+".png"
        graph_path = "graph/scene_" + str(i)+".png"
        

        cars[0].position = mi.Point3f(robot1_pos.x, robot1_pos.y, robot1_pos.z)
        cars[0].look_at(mi.Point3f(robot1_orien.x, robot1_orien.y, robot1_orien.z))

        # cars[1].position = mi.Point3f(robot2_pos.x, robot2_pos.y, robot2_pos.z)
        # cars[1].look_at(mi.Point3f(robot2_orien.x, robot2_orien.y, robot2_orien.z))

       
        scene.get("rx").position = mi.Point3f(robot1_pos.x+0.2, robot1_pos.y, robot1_pos.z)
        # scene.update()
        
        paths = p_solver(scene, max_depth=5)
        a, tau = paths.cir(normalize_delays=True, out_type="numpy")

        print("Shape of a: ", a.shape)
        print("Shape of tau: ", tau.shape)

        t = tau[0,0,:]/1e-9 # Scale to ns
        a_abs = np.abs(a)[0,0,0,0,:,0]
        a_max = np.max(a_abs)

        # And plot the CIR
        plt.figure()
        plt.title("Channel impulse response")
        plt.stem(t, a_abs)
        plt.xlabel(r"$\tau$ [ns]")
        plt.ylabel(r"$|a|$");
        plt.savefig(graph_path)
        plt.close()

        # plt.show()

        #scene.preview(paths=paths);
        # scene.render(camera=my_cam, paths=paths, clip_at=20);
        if not no_preview:
            if(i%1==0):
                
                scene.render_to_file(camera=my_cam,
                         filename=img_path,
                         resolution=[650,500],
                         paths =paths );
    
                scene.render_to_file(camera=my_cam2,
                         filename=img_path2,
                         resolution=[650,500],
                         paths =paths);
        


        
        # canvas.update()
        # canvas.app.process_events()

       
        

        


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass