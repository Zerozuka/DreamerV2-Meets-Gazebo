#!/usr/bin/env python3
"""Gazebo Harmonic の world を gz-transport 経由で制御する。

Gazebo Classic では gazebo_ros が下記の ROS サービスを出していたが、
Harmonic にはこれらが存在しない。gazebo_msgs パッケージ自体 ros_gz には
含まれない。

    /gazebo/pause_physics            -> /world/<name>/control の pause=True
    /gazebo/unpause_physics          -> /world/<name>/control の pause=False
    /gazebo/reset_simulation         -> /world/<name>/control の reset.all
    /gazebo/set_model_state          -> /world/<name>/set_pose
    /gazebo/get_physics_properties   -> 廃止。max_step_size は world SDF の
                                        設定値なのでコンストラクタで受け取る

ros_gz_bridge を経由せず gz-transport を直接叩いている。強化学習の reset は
エピソードごとに呼ばれるため bridge の往復を挟まない方が速く、ROS のノードを
立てずに使えるので gymnasium.Env の中から呼びやすい。

world 側に次のシステムプラグインが必要。jetbot_tellus.launch.py が読む
world には入れてある。

    gz::sim::systems::UserCommands     set_pose を提供
    gz::sim::systems::SceneBroadcaster control を提供

使用例:

    gz = GzWorldControl(world="default")
    gz.pause()
    gz.set_pose("jetbot_1", x=5.16, y=7.77, z=0.1, yaw=-1.57)
    gz.unpause()
"""

import math

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.world_control_pb2 import WorldControl
from gz.transport13 import Node

#: world SDF の <max_step_size>。tellus3.world / tellus3_with_road.world とも
#: 0.001 を指定している。Classic では /gazebo/get_physics_properties で実行時に
#: 取得していたが、SDF にハードコードされた設定値なので定数で持つ方が単純。
DEFAULT_MAX_STEP_SIZE = 0.001


class GzWorldControlError(RuntimeError):
    """gz-transport の呼び出しが失敗した。"""


class GzWorldControl:

    def __init__(self, world="default", timeout_ms=3000,
                 max_step_size=DEFAULT_MAX_STEP_SIZE):
        self._node = Node()
        self.world = world
        self.timeout_ms = timeout_ms
        self.max_step_size = max_step_size
        self._control_srv = f"/world/{world}/control"
        self._set_pose_srv = f"/world/{world}/set_pose"

    # ---- 内部ヘルパ ------------------------------------------------------

    def _request(self, service, req, req_type):
        ok, rep = self._node.request(
            service, req, req_type, Boolean, self.timeout_ms
        )
        if not ok:
            raise GzWorldControlError(
                f"{service} の呼び出しが失敗した。world 名 '{self.world}' が"
                f"合っているか、必要なシステムプラグイン "
                f"(UserCommands / SceneBroadcaster) が world に入っているかを"
                f"確認すること。"
            )
        if not rep.data:
            raise GzWorldControlError(f"{service} が false を返した")
        return rep

    # ---- 物理の一時停止 / 再開 -------------------------------------------

    def pause(self):
        """物理を止める。Classic の /gazebo/pause_physics に相当。"""
        req = WorldControl()
        req.pause = True
        self._request(self._control_srv, req, WorldControl)

    def unpause(self):
        """物理を再開する。Classic の /gazebo/unpause_physics に相当。"""
        req = WorldControl()
        req.pause = False
        self._request(self._control_srv, req, WorldControl)

    def step(self, n=1):
        """一時停止中に n ステップだけ進める。

        Classic には無かったが、RL では観測を取る前に決まったステップ数だけ
        進めたい場面があるので用意した。
        """
        req = WorldControl()
        req.pause = True
        req.multi_step = n
        self._request(self._control_srv, req, WorldControl)

    # ---- リセット --------------------------------------------------------

    def reset_all(self):
        """world 全体を初期状態に戻す。Classic の /gazebo/reset_simulation。

        すべてのモデルが SDF の初期姿勢に戻り、sim time も 0 に戻る。

        注意: **実行時に spawn したモデルは削除される。** world SDF に書かれて
        いないものは「初期状態」に存在しないためである。launch で
        ros_gz_sim create でロボットを置いた場合、これを呼ぶとロボットが
        world から消える (実測で確認済み)。

        したがって強化学習のエピソード reset には使えない。ロボットを初期位置に
        戻したいだけなら set_pose を使うこと。
        """
        req = WorldControl()
        req.reset.all = True
        self._request(self._control_srv, req, WorldControl)

    def reset_time_only(self):
        """sim time だけ 0 に戻す。モデルの姿勢は変えない。"""
        req = WorldControl()
        req.reset.time_only = True
        self._request(self._control_srv, req, WorldControl)

    # ---- 姿勢の設定 ------------------------------------------------------

    def set_pose(self, name, x=0.0, y=0.0, z=0.0,
                 roll=0.0, pitch=0.0, yaw=0.0):
        """モデルを指定の姿勢へ移す。Classic の /gazebo/set_model_state。

        Classic の SetModelState は速度も指定できたが、Harmonic の set_pose は
        姿勢のみ。速度を 0 に戻したい場合は reset_all を使うか、DiffDrive に
        cmd_vel で 0 を送る。
        """
        req = Pose()
        req.name = name
        req.position.x = float(x)
        req.position.y = float(y)
        req.position.z = float(z)

        qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)
        req.orientation.x = qx
        req.orientation.y = qy
        req.orientation.z = qz
        req.orientation.w = qw

        self._request(self._set_pose_srv, req, Pose)


def quaternion_from_euler(roll, pitch, yaw):
    """ZYX 順のオイラー角からクォータニオン (x, y, z, w) を作る。

    tf_transformations.quaternion_from_euler と同じ規約 (sxyz)。この 1 関数の
    ために ROS への依存を増やしたくないので内製した。
    """
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


if __name__ == "__main__":
    # 動作確認用。Gazebo が起動している状態で実行する。
    import sys

    gz = GzWorldControl(world=sys.argv[1] if len(sys.argv) > 1 else "default")
    print("pause");    gz.pause()
    print("step(10)"); gz.step(10)
    print("unpause");  gz.unpause()
    print("OK")
