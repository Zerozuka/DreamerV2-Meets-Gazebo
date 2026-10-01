#!/usr/bin/env python3
"""学習済み重み・データセット・出力先のパスを環境変数で解決する。

移植前は開発者の home を指す絶対パスが 90 箇所ハードコードされていた。

    /home/icon-group/catkin_ws/src/i_jepa/control_jepa/test/results/...
    /home/icon-group/Documents/Josh/sionna/Tellus/sionna_test/...
    /home/icon-group/image_token/vit_sequence/results/

これらは clone しただけの環境には存在しないため、別マシンに持っていくと
動かなかった。種類ごとに基準ディレクトリを環境変数で与える方式に変えた。

    JEPA_MODEL_DIR    学習済み重みの置き場 (既定: ./models)
    JEPA_DATA_DIR     データセットの置き場 (既定: ./data)
    JEPA_OUTPUT_DIR   実行結果の出力先     (既定: ./output)

リポジトリに同梱されている資産 (Sionna のシーン、メッシュ) は ament_index で
解決するため、ここでは扱わない。

使用例:

    from paths import model_path, output_path

    MODEL_PATH = model_path("results/CarRacing-v2_0_pomdp/20_dec_gazebo",
                            "models_best_8.pth")
    log = output_path("predicted_power_log.csv")
"""

import os


def _base(env_var, default):
    return os.environ.get(env_var, os.path.join(os.getcwd(), default))


def model_path(*parts):
    """学習済み重みのパス。JEPA_MODEL_DIR が基準。

    存在しない場合の扱いは呼び出し側に任せる。train_gazebo.py のように
    os.path.exists で分岐してスクラッチ学習に落ちる実装があるため、
    ここでは例外にしない。必須なら require() を併用する。
    """
    return os.path.join(_base("JEPA_MODEL_DIR", "models"), *parts)


def data_path(*parts):
    """データセットのパス。JEPA_DATA_DIR が基準。"""
    return os.path.join(_base("JEPA_DATA_DIR", "data"), *parts)


def output_path(*parts):
    """出力先のパス。JEPA_OUTPUT_DIR が基準。

    親ディレクトリを作ってから返すので、呼び出し側で mkdir しなくてよい。
    """
    p = os.path.join(_base("JEPA_OUTPUT_DIR", "output"), *parts)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    return p


def require(path, description):
    """読み込み必須のファイルを検査する。無ければ何を用意すべきかを示す。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{description} が見つからない: {path}\n"
            f"  JEPA_MODEL_DIR / JEPA_DATA_DIR を設定するか、"
            f"該当ファイルを配置すること。\n"
            f"  学習済み重みはリポジトリに含まれておらず、著者に照会中である。"
        )
    return path
