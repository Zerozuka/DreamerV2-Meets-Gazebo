#!/usr/bin/env python3
"""Sionna 2.x を使うための補助。バックエンド選択と CIR の形状合わせ。

このリポジトリのコードは Sionna RT 1.x 系の API で書かれていた。RT の API
(load_scene / PathSolver / RadioMapSolver / SceneObject / ITURadioMaterial /
scene.edit) は 2.x にそのまま引き継がれているので大半は無改修で動くが、
実際に動かして 2 点の非互換が見つかった。

1. バックエンドの既定が OptiX (GPU) になった
--------------------------------------------
Sionna RT 2.x は import 時に mitsuba の variant を cuda_ad_mono_polarized に
設定する。OptiX のライブラリ (libnvoptix.so.1) が無い環境では load_scene() が
次で落ちる。

    RuntimeError: [parser.cpp:1718] failed to instantiate scene plugin of
    type "scene": Could not initialize OptiX!

NVIDIA の GPU とドライバがあっても libnvoptix.so.1 は別途必要である。
select_backend() を sionna.rt の import より前に呼ぶと、OptiX が使えない
環境では LLVM (CPU) にフォールバックする。

2. paths.cir() の戻り値が cir_to_ofdm_channel にそのまま渡せない
----------------------------------------------------------------
1.x 系のコードは次のように書かれていた。

    a, tau = paths.cir(normalize_delays=True, out_type="numpy")
    h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=False)

2.x ではこれが通らない。out_type ごとに実測した結果は次のとおり。

    out_type          a の型     tau の型     cir_to_ofdm_channel
    既定 ("drjit")    list       TensorXf     NG (dtype 属性がない)
    "numpy"           ndarray    ndarray      NG (dim 属性がない)
    "torch"           Tensor     Tensor       NG (次元不足)

PHY が PyTorch 化されたので torch テンソルを要求するようになり、さらに
バッチ次元が必要になった。

    paths.cir が返す形状         cir_to_ofdm_channel が期待する形状
    a:   (1,1,1,1,53,1)  6 次元   [batch,num_rx,num_rx_ant,num_tx,num_tx_ant,
                                   num_paths,num_time_steps]  7 次元
    tau: (1,1,53)        3 次元   [batch,num_rx,num_tx,num_paths]  4 次元

cir_for_ofdm() が out_type="torch" で取得してバッチ次元を足す。
"""

import os


def select_backend(prefer=None, verbose=True):
    """mitsuba の variant を決める。sionna.rt を import する前に呼ぶこと。

    prefer に variant 名を渡すとそれを使う。省略時は環境変数
    SIONNA_MI_VARIANT、それも無ければ OptiX が使えるかを実際に試して
    cuda 系か llvm 系を選ぶ。

    戻り値は選ばれた variant 名。
    """
    import mitsuba as mi

    if prefer is None:
        prefer = os.environ.get("SIONNA_MI_VARIANT")

    if prefer:
        mi.set_variant(prefer)
        if verbose:
            print(f"[sionna_compat] mitsuba variant = {mi.variant()} (指定)")
        return mi.variant()

    # OptiX が使えるかを実際に試す。variant を設定するだけでは失敗せず、
    # シーンを作る段で落ちるので、最小のシーンで確かめる。
    for cand in ("cuda_ad_mono_polarized", "llvm_ad_mono_polarized"):
        if cand not in mi.variants():
            continue
        try:
            mi.set_variant(cand)
            mi.load_dict({"type": "scene"})
        except Exception as e:
            if verbose:
                print(f"[sionna_compat] {cand} は使えない ({type(e).__name__})")
            continue
        if verbose:
            print(f"[sionna_compat] mitsuba variant = {mi.variant()}")
        return mi.variant()

    raise RuntimeError(
        "使える mitsuba の variant が見つからない。"
        f"利用可能: {mi.variants()}"
    )


def cir_for_ofdm(paths, **kwargs):
    """cir_to_ofdm_channel にそのまま渡せる (a, tau) を返す。

    1.x 系のコードは out_type="numpy" で取って渡していたが、2.x では
    torch テンソルかつバッチ次元付きが必要になった。

    kwargs は paths.cir にそのまま渡す (normalize_delays など)。
    out_type は "torch" に固定するので指定しないこと。
    """
    kwargs.pop("out_type", None)
    a, tau = paths.cir(out_type="torch", **kwargs)
    # 先頭にバッチ次元を足す。
    #   a:   [num_rx,num_rx_ant,num_tx,num_tx_ant,num_paths,num_time_steps]
    #     -> [1, ...]
    #   tau: [num_rx,num_tx,num_paths] -> [1, ...]
    return a.unsqueeze(0), tau.unsqueeze(0)
