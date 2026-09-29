from setuptools import setup

package_name = 'wireless_jepa'

# Python モジュールをまだインストールしない理由
# ------------------------------------------------
# src/ には dreamerv2/ と utils/ があるが、control_jepa/test/ にも
# 同名のトップレベルパッケージが存在する。dreamerv2/ は両者でバイト単位
# まで同一のコピーで、utils/ も同名。どちらも site-packages に入れると
# トップレベル名が衝突する。
#
# 重複の解消 (共通パッケージへの集約) は移植後の整理項目なので、
# それまでは packages=[] とし、スクリプトは README のとおり
# パス指定で実行する。rclpy 移植 (#3) で console_scripts を登録する。
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
    description='Wireless-JEPA。Sionna RT の CSI からチャネル表現を学習する',
    license='TODO',
    entry_points={
        'console_scripts': [],
    },
)
