from setuptools import setup

package_name = 'control_jepa'

# このファイルについて
# --------------------
# 元は upstream の dreamerv2 (RajGhugare19/dreamerv2) から持ち込まれた
# setup.py で、name="dreamerv2" のまま残っていた。find_packages() は
# このディレクトリ直下に __init__.py を持つパッケージがないため何も拾わず、
# 実質的に機能していなかった。ament_python パッケージとして作り直す。
#
# Python モジュールをまだインストールしない理由
# ------------------------------------------------
# test/ には dreamerv2/, utils/, wutils/, datasets/, dataset/ があるが、
# dreamerv2/ と utils/ は wireless_jepa/src/ にも同名で存在する
# (dreamerv2/ はバイト単位まで同一のコピー)。どちらも site-packages に
# 入れるとトップレベル名が衝突する。
#
# あわせて、本番コードが test/ 配下に置かれている点も整理対象。
# 重複解消とディレクトリ構成の是正は移植後の整理項目なので、それまでは
# packages=[] とし、スクリプトは README のとおりパス指定で実行する。
# rclpy 移植 (#3) で console_scripts を登録する。
setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='icon-group',
    maintainer_email='icon-group@todo.todo',
    description='Control-JEPA。Gazebo の観測とロボット動力学から制御表現を学習する',
    license='TODO',
    entry_points={
        'console_scripts': [],
    },
)
