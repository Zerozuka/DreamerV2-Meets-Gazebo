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

シミュレーション速度について
----------------------------
実時間係数を落とす要因は 2 段あり、両方を実測で切り分けた。

第 1 段: 衝突形状 (解消済み)

    jetbot_real.urdf.xacro は 20 個すべてのリンクで視覚用の高精細 STL を
    そのまま衝突形状にも使っていた (collision の三角形 合計 745,410、
    nano_link.STL 単体で 374,540)。DART が毎ステップこれを衝突判定するため、
    ロボットを spawn した時点で 1.0 から 0.26 に落ちていた。

    接地に関わるのは車輪のタイヤとキャスタだけで、他は固定関節で base_link に
    まとめられる。境界ボックスから箱・円柱・球に置き換えて約 600 三角形にした。
    これで gz 単体では 1.000 が出る。

第 2 段: センサのブリッジ (現在の支配要因)

    衝突形状を直した後の実測値。

        bridge_sensors:=true    real_time_factor 0.0025
        bridge_sensors:=false   real_time_factor 1.000

    ros_gz_bridge が /image_raw2 と /scan を購読すると Sensors システムが
    描画を始める。GPU が使えない環境 (LIBGL_ALWAYS_SOFTWARE=1) では
    640x480 のカメラ 30 Hz と 1147 点 LiDAR を CPU で描くため 400 倍遅くなる。

    走行や odom の確認だけなら bridge_sensors:=false で等速で回る。
    C-JEPA の学習には画像が必要なので true にするが、その場合は GPU が
    使える環境を用意すること。
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    LaunchConfiguration,
    PathJoinSubstitution,
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
        # カメラと LiDAR を ROS 側へ橋渡しするか。
        #
        # これを購読すると Sensors システムが描画を始めるため、GPU が使えない
        # 環境では実時間係数が 1.000 から 0.0025 まで落ちる (実測)。走行や odom の
        # 確認だけなら false にすると等速で回る。詳細はモジュール docstring 参照。
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

    # サーバは gui の値に関わらず常に -s (サーバのみ) で起動し、GUI が要るときは
    # 別プロセスとして後から繋ぐ。1 プロセスで両方やらせると実時間係数が 14 倍
    # 落ちるためである。
    #
    # 原因は DISPLAY である。DISPLAY が設定されていると、サーバ側のカメラセンサ
    # の描画までが EGL ではなく GLX を選ぶ。リモートデスクトップ (NoMachine /
    # VNC) や SSH の X 転送の X サーバはハードウェア GLX を持たないので、そこで
    # ソフトウェア描画に落ち、物理演算まで道連れになる。実測値は次のとおり。
    #
    #     1 プロセス (gz sim -r, DISPLAY あり)     RTF 0.068  GPU 0 %
    #     サーバのみ (gz sim -s -r, DISPLAY なし)  RTF 0.989  GPU 33 %
    #     サーバ + GUI を別プロセス                RTF 0.963  GPU 34 %
    #
    # --headless-rendering はサーバの描画を EGL に固定する指定で、DISPLAY が
    # 設定されていてもセンサ描画が GPU に載る。GUI ウィンドウ自体は X サーバの
    # GLX を使うので、リモートデスクトップではソフトウェア描画のままだが、
    # 別プロセスなので物理演算は巻き込まれない。
    gz_flags = '-s -r -v 2 --headless-rendering '

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

    # gui:=true のときだけ GUI を別プロセスで立てる。サーバが gz-transport の
    # サービスを出すまで少し待つ (即座に起動すると接続に失敗することがある)。
    gz_gui = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=['gz', 'sim', '-g'],
                output='screen',
                condition=IfCondition(LaunchConfiguration('gui')),
            )
        ],
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
            gz_gui,
            robot_state_publisher,
            spawn,
            bridge,
            sensor_bridge,
            result_image_viewer,
        ]
    )
