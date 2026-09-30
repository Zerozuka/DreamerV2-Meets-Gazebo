#!/usr/bin/env python3
"""sensor_msgs/Image を OpenCV 形式 (bgr8) の numpy 配列に変換する。

cv_bridge の代替である。cv_bridge を捨てた理由は NumPy のバージョンである。

ROS 2 Jazzy の cv_bridge は apt で入る C++ 拡張 (cv_bridge_boost.so) で、
NumPy 1 系の C API に対してコンパイルされている。NumPy 2 系の下では拡張の
初期化が内部で失敗し、変換テーブル (cvtype_to_name) が空のまま残るため、
実行時に次で落ちる。

    KeyError: 16
      File ".../cv_bridge/core.py", line 276, in cv2_to_imgmsg
        if self.cvtype_to_name[self.encoding_to_cvtype2(encoding)] != cv_type:

apt 由来のバイナリなので pip では差し替えられない。一方 Sionna 2.x は
numpy>=2.2.6 を要求する。cv_bridge を残すと環境を NumPy 1 と NumPy 2 の
2 つに割らざるを得ず、rclpy と sionna を同一プロセスで使うスクリプト
(sionna_pos.py, wireless_jepa.py, channel_generate.py) がどちらでも動かなく
なる。

このリポジトリでの cv_bridge の使い方は全 55 箇所すべてが

    bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

の一形だけだったので、その 1 経路を自前で実装して cv_bridge への依存を
外した。rclpy と全メッセージ型は NumPy 2 でそのまま動く (実測済み) ため、
これで環境が 1 つにまとまる。
"""

import numpy as np

# エンコーディングごとのチャンネル数。このリポジトリで出てくる 8 bit 系だけを
# 扱う。Gazebo Harmonic のカメラは ros_gz bridge 経由で rgb8 を流す。
_CHANNELS = {
    "mono8": 1,
    "8UC1": 1,
    "bgr8": 3,
    "rgb8": 3,
    "8UC3": 3,
    "bgra8": 4,
    "rgba8": 4,
    "8UC4": 4,
}


def imgmsg_to_bgr8(msg):
    """sensor_msgs/Image を (height, width, 3) の bgr8 配列にして返す。

    cv_bridge の imgmsg_to_cv2(msg, desired_encoding='bgr8') と同じ結果を返す。
    戻り値は連続かつ書き込み可能な新しい配列なので、そのまま cv2 に渡せる
    (msg.data のバッファは read-only であり、スライスによる反転はストライドが
    負になって cv2 が受け付けないことがあるため、必ずコピーして返す)。

    cv_bridge との差が 1 点だけある。cv_bridge は行末パディングを無視して
    shape=(height, step/n_channels, n_channels) で buffer を読むため、
    step > width * channels のとき行が少しずつずれた画像を返す。ここでは
    step を尊重して width 分だけ切り出すので、その場合は結果が異なる。
    Gazebo Harmonic の ros_gz bridge は step == width * 3 で流してくるので
    実運用では差は出ない。パディングなしの rgb8 / bgr8 / mono8 / rgba8 /
    bgra8 については cv_bridge と完全に一致することを実測で確認済みである。
    """
    encoding = msg.encoding
    try:
        channels = _CHANNELS[encoding]
    except KeyError:
        raise ValueError(
            f"未対応のエンコーディング: {encoding!r}。"
            f"対応しているのは {sorted(_CHANNELS)}。"
            "必要なら ros_image.py の _CHANNELS に追加すること。"
        ) from None

    expected = msg.step * msg.height
    if len(msg.data) < expected:
        raise ValueError(
            f"Image のデータが足りない: {len(msg.data)} バイトだが "
            f"step({msg.step}) * height({msg.height}) = {expected} バイト必要。"
        )

    # msg.step は 1 行あたりのバイト数で、width * channels より大きいことが
    # ある (行末のパディング)。step で切ってから width 分だけ取る。
    flat = np.frombuffer(msg.data, dtype=np.uint8, count=expected)
    rows = flat.reshape(msg.height, msg.step)
    img = rows[:, : msg.width * channels].reshape(msg.height, msg.width, channels)

    if encoding in ("bgr8", "8UC3"):
        out = img
    elif encoding == "rgb8":
        out = img[:, :, ::-1]
    elif encoding in ("bgra8", "8UC4"):
        out = img[:, :, :3]
    elif encoding == "rgba8":
        out = img[:, :, 2::-1]
    else:  # mono8 / 8UC1 -- cv_bridge の GRAY2BGR と同じく 3 チャンネルに複製する
        out = np.repeat(img, 3, axis=2)

    return np.ascontiguousarray(out)
