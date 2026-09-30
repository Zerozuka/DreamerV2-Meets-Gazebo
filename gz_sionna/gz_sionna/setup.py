import os

from setuptools import setup

package_name = 'gz_sionna'


def data_files_in(*dirs):
    """dirs 以下を再帰的にたどり share/<pkg>/ に同じ階層で配置する。

    ament_python には ament_cmake の install(DIRECTORY ...) に相当する仕組みが
    ないため、ディレクトリごとにファイル一覧を組み立てる。
    """
    entries = []
    for d in dirs:
        for root, subdirs, files in os.walk(d):
            # __pycache__ は install 対象にしない。スクリプトをその場で実行すると
            # 生成され、--symlink-install で
            #   error: [Errno 17] File exists: .../__pycache__/*.pyc
            # となってビルドが落ちる。
            subdirs[:] = [s for s in subdirs if s != '__pycache__']
            paths = [os.path.join(root, f) for f in files
                     if not f.endswith(('.pyc', '.pyo'))]
            if paths:
                entries.append((os.path.join('share', package_name, root), paths))
    return entries


setup(
    name=package_name,
    version='0.0.0',
    # src/ 配下のノードはまだ rospy ベースで、rclpy 移植 (#3) の対象。
    # モジュールとしての登録はそこで行う。
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ] + data_files_in(
        # config/ には GazeboEnv が読むコースデータ 2 件が入っている。
        # path_points.csv (x, y, yaw) は road_model.sdf の road_section_* の
        # pose と 174 行すべて一致しており、同じコースを指している。
        # cross_markers_400.csv (x1, y1, x2, y2) は進捗と報酬の基準。
        'config',
        'launch',
        # models/with_materials/ には Sionna RT 用の Mitsuba シーン
        # (untitled.xml) と ITU マテリアル名付きの .ply が 133 個入っている。
        'models',
        'rviz',
        # src/gz_world_control.py を他パッケージ (control_jepa) から使うため
        # share にも置く。本来は Python モジュールとして入れたいが、
        # dreamerv2 / utils の名前衝突を解消するまでは packages=[] のため、
        # 呼び出し側は share のパスを ament_index 経由で解決して import する。
        'src',
        'worlds',
    ),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='icon-group',
    maintainer_email='icon-group@todo.todo',
    description='Gazebo と Sionna RT を同期させるノードと、対応する world・モデル資産',
    license='TODO',
    entry_points={
        # ノードの console_scripts は rclpy 移植 (#3) で登録する。
        'console_scripts': [],
    },
)
