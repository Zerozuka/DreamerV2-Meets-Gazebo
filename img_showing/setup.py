import os

from setuptools import setup

package_name = 'img_showing'


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
    # src/img_show.py はまだ rospy ベースで、rclpy 移植 (#3) の対象。
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ] + data_files_in('launch'),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='icon-group',
    maintainer_email='icon-group@todo.todo',
    description='結果画像を表示するビューアノード',
    license='TODO',
    entry_points={
        # ノードの console_scripts は rclpy 移植 (#3) で登録する。
        'console_scripts': [],
    },
)
