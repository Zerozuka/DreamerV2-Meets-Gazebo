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
        for root, _, files in os.walk(d):
            paths = [os.path.join(root, f) for f in files]
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
        'launch',
        # models/with_materials/ には Sionna RT 用の Mitsuba シーン
        # (untitled.xml) と ITU マテリアル名付きの .ply が 133 個入っている。
        'models',
        'rviz',
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
