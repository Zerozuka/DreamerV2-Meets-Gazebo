"""Tellus アリーナに Jetbot を置く。ROS 2 Jazzy + Gazebo Harmonic 版。

ROS 1 の jetbot_tellus.launch からの移植。対応は次のとおり。

    $(find pkg)                                -> FindPackageShare('pkg')
    gazebo_ros/launch/empty_world.launch       -> ros_gz_sim/launch/gz_sim.launch.py
    gazebo_ros spawn_model                     -> ros_gz_sim の create
    <param name="robot_description" command=>  -> robot_state_publisher のパラメータ
    (Classic は ROS と直結)                     -> ros_gz_bridge でトピックを橋渡し

使い方:

    ros2 launch gz_sionna jetbot_tellus.launch.py
    ros2 launch gz_sionna jetbot_tellus.launch.py gui:=false
    ros2 launch gz_sionna jetbot_tellus.launch.py world:=tellus3_with_road.world

world の既定値は tellus3.world。README が使っていた
tellus3_with_road.world は cross_line / race_end / receiver_1..3 /
road_model の 6 モデルが未入手で読み込めないため、既定にしていない。

仮想ディスプレイ (NoMachine 等) で使う場合は、起動前に次の環境変数が必要。
Sensors システムは描画コンテキストを要求するため、設定しないと segfault する。

    export LIBGL_ALWAYS_SOFTWARE=1
    export QT_QPA_PLATFORM=xcb
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    gz_sionna_share = get_package_share_directory('gz_sionna')
    jetbot_world_share = get_package_share_directory('jetbot_world')

    args = [
        DeclareLaunchArgument('world', default_value='tellus3.world'),
        DeclareLaunchArgument('robot_name', default_value='jetbot_1'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('gui', default_value='true'),
        # ROS 1 版と同じ初期姿勢
        DeclareLaunchArgument('x_pos', default_value='5.163443'),
        DeclareLaunchArgument('y_pos', default_value='7.766847'),
        DeclareLaunchArgument('z_pos', default_value='0.1'),
        DeclareLaunchArgument('yaw', default_value='-1.573762'),
        DeclareLaunchArgument('view_result_image', default_value='false'),
        # カメラと LiDAR を ROS 側へ橋渡しするか。ROS 側の負荷を下げたい場合に
        # false にする。
        #
        # 注意: これを false にしてもシミュレーションの速度は改善しない。
        # GPU が使えない環境 (LIBGL_ALWAYS_SOFTWARE=1) で計測した実時間係数は
        # 次のとおりで、ブリッジの有無で変わらなかった。
        #
        #   bridge_sensors:=true   real_time_factor 0.0086 - 0.017
        #   bridge_sensors:=false  real_time_factor 0.0085
        #   /clock  12 - 15 Hz (本来 1000 Hz)
        #   /imu    0.43 Hz    (本来 30 Hz)
        #
        # world の Sensors システムが購読者の有無にかかわらず描画している
        # ためと思われるが、未確認。一方 ROS を介さず gz へ直接指令を送った
        # 場合は 6 秒で 1.767 m 走行できており、この環境でも条件次第では
        # 実用速度が出る。C-JEPA の学習には GPU が使える環境を用意すること。
        DeclareLaunchArgument('bridge_sensors', default_value='true'),
    ]

    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')

    # ---- リソースパス --------------------------------------------------
    # model://tellus, model://cube などを解決するために gz_sionna の models を、
    # ロボットのメッシュのために jetbot_world の share の親を通す。URDF の
    # package://jetbot_world/meshes/... は sdformat により
    # model://jetbot_world/meshes/... に変換されるため、model:// の探索先として
    # jetbot_world ディレクトリを含むパス (= share の親) が必要になる。
    resource_paths = [
        AppendEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH', os.path.join(gz_sionna_share, 'models')
        ),
        AppendEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH', os.path.dirname(jetbot_world_share)
        ),
    ]

    # ---- Gazebo -------------------------------------------------------
    world_path = PathJoinSubstitution(
        [gz_sionna_share, 'worlds', LaunchConfiguration('world')]
    )

    # gui:=false ならサーバのみ (-s)。-r は即座に走らせる指定。
    gz_flags = PythonExpression([
        "'-r -v 2 ' if '", LaunchConfiguration('gui'),
        "'.lower() in ('true', '1') else '-s -r -v 2 '",
    ])

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py']
            )
        ),
        launch_arguments={
            'gz_args': [gz_flags, world_path],
            'on_exit_shutdown': 'true',
        }.items(),
    )

    # ---- ロボット記述 --------------------------------------------------
    # ParameterValue で value_type=str を明示しないと、launch_ros が xacro の
    # 出力 (URDF の XML) を YAML として解釈しようとして次のエラーになる。
    #   Unable to parse the value of parameter robot_description as yaml
    robot_description = ParameterValue(
        Command([
            'xacro ',
            PathJoinSubstitution(
                [jetbot_world_share, 'urdf', 'jetbot_real.urdf.xacro']
            ),
            ' botname:=', robot_name,
        ]),
        value_type=str,
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time,
        }],
    )

    # ---- spawn ---------------------------------------------------------
    # Classic の gazebo_ros spawn_model に相当。robot_description トピックから
    # 読ませるので、robot_state_publisher が先に上がっている必要がある。
    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_jetbot',
        output='screen',
        arguments=[
            '-topic', 'robot_description',
            '-name', robot_name,
            '-x', LaunchConfiguration('x_pos'),
            '-y', LaunchConfiguration('y_pos'),
            '-z', LaunchConfiguration('z_pos'),
            '-Y', LaunchConfiguration('yaw'),
        ],
    )

    # ---- ros_gz_bridge -------------------------------------------------
    # トピック名は gazebo_env.py が購読・配信しているものに合わせてある
    # (cmd_vel / odom / image_raw2)。名前を変えると Python 側の修正が必要。
    #
    # 記法: <topic>@<ROS 型>@<gz 型>  双方向
    #       <topic>@<ROS 型>[<gz 型>  gz から ROS へ
    #       <topic>@<ROS 型>]<gz 型>  ROS から gz へ
    # 描画を伴わないもの。常に橋渡しする。
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_bridge',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
        ],
    )

    # 描画を伴うもの。購読すると Sensors システムが回り続けるので分離した。
    sensor_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_sensor_bridge',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[
            '/image_raw2@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
        ],
        condition=IfCondition(LaunchConfiguration('bridge_sensors')),
    )

    # ---- 結果画像ビューア ------------------------------------------------
    # ROS 1 版は pkg="jepa_world" type="img_show.py" を起動していたが、
    # jepa_world はリポジトリに存在しない。img_showing の img_show.py が
    # 同名・同内容と思われるためそちらを使う (著者に確認中)。
    # /result_img を publish する側がまだ移植されていないので既定は false。
    result_image_viewer = Node(
        package='img_showing',
        executable='img_show.py',
        name='camera_view_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('view_result_image')),
    )

    # ROS 1 版はここで wireless_jepa の wireless_jepa.py も起動していたが、
    # まだ rospy ベースなので含めていない。rclpy 移植後に追加する。

    return LaunchDescription(
        args
        + resource_paths
        + [
            gz_sim,
            robot_state_publisher,
            spawn,
            bridge,
            sensor_bridge,
            result_image_viewer,
        ]
    )
