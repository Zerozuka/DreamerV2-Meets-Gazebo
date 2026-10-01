#!/usr/bin/env python3
"""Check that the workspace is set up correctly, one component at a time.

Run it with the virtual environment's interpreter, after sourcing both ROS 2
and the workspace:

    source /opt/ros/jazzy/setup.bash
    source ~/ros2_ws/install/setup.bash
    cd ~/ros2_ws/src/DreamerV2-Meets-Gazebo
    .venv/bin/python tools/verify_setup.py

Every check prints OK or NG. A failing check says what to do about it and
points at the section of README.md that explains why. The exit code is 0 only
when everything passed, so this is usable from a script.

Pass --quick to skip the ray tracing check, which is the slow one.
"""

import argparse
import importlib.metadata as md
import os
import sys

RESULTS = []


def check(label, hint=""):
    """Decorator that runs one check immediately and records the outcome.

    The wrapped function returns a string to print next to OK. Raising means
    the check failed; `hint` is then shown as the thing to do about it.
    """

    def wrap(fn):
        try:
            detail = fn()
            print(f"  OK  {label}" + (f"  {detail}" if detail else ""))
            RESULTS.append((label, True))
        except Exception as exc:  # noqa: BLE001 - 何が出ても報告して次へ進む
            first = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            print(f"  NG  {label}")
            print(f"      {type(exc).__name__}: {first[:160]}")
            if hint:
                print(f"      -> {hint}")
            RESULTS.append((label, False))
        return fn

    return wrap


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--quick", action="store_true",
                    help="skip the ray tracing check")
args = parser.parse_args()

print("=== interpreter ===")


@check("python 3.12 from the virtual environment",
       "create it with: uv venv --system-site-packages "
       "--python /usr/bin/python3.12 .venv  (README: step 3)")
def _python():
    version = sys.version.split()[0]
    if sys.prefix == sys.base_prefix:
        raise RuntimeError(
            f"not running inside a virtual environment (python {version}). "
            "Use .venv/bin/python, not python3"
        )
    if not version.startswith("3.12"):
        raise RuntimeError(
            f"python {version}, expected 3.12 to match the apt ROS 2 build")
    return f"python {version}"


print("\n=== ROS 2 (from apt, visible through --system-site-packages) ===")


@check("rclpy",
       "source /opt/ros/jazzy/setup.bash, and make sure .venv was created "
       "with --system-site-packages  (README: steps 2 and 3)")
def _rclpy():
    import rclpy  # noqa: F401
    return ""


@check("message packages",
       "same as rclpy: source ROS 2 first")
def _msgs():
    from geometry_msgs.msg import Point, Twist  # noqa: F401
    from nav_msgs.msg import Odometry  # noqa: F401
    from sensor_msgs.msg import Image  # noqa: F401
    from std_msgs.msg import Float32MultiArray, Int32  # noqa: F401
    return ""


@check("tf_transformations",
       "install the pip build of transforms3d: the apt one calls "
       "np.maximum_sctype, removed in NumPy 2  (README: The NumPy 2 problem)")
def _tf():
    import tf_transformations  # noqa: F401
    return ""


@check("cv2",
       "install opencv-python from pip: the apt build is compiled against "
       "the NumPy 1 C API  (README: The NumPy 2 problem)")
def _cv2():
    import cv2
    return f"opencv {cv2.__version__}"


print("\n=== workspace (needs colcon build and a sourced install) ===")


@check("gz_sionna share directory",
       "build and source the workspace: colcon build --symlink-install in "
       "~/ros2_ws, then source ~/ros2_ws/install/setup.bash "
       "(README: steps 5 and 6)")
def _share():
    from ament_index_python.packages import get_package_share_directory
    path = get_package_share_directory("gz_sionna")
    sys.path.insert(0, os.path.join(path, "src"))
    return path


@check("ros_image (the cv_bridge replacement)",
       "rebuild the workspace so gz_sionna/src is installed")
def _ros_image():
    import numpy as np
    from ros_image import imgmsg_to_bgr8
    from sensor_msgs.msg import Image

    msg = Image()
    msg.height, msg.width, msg.encoding, msg.step = 4, 6, "rgb8", 18
    rgb = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    msg.data = rgb.tobytes()
    bgr = imgmsg_to_bgr8(msg)
    if not np.array_equal(bgr, rgb[:, :, ::-1]):
        raise RuntimeError("the red and blue channels were not swapped")
    return f"{bgr.shape} bgr8"


print("\n=== Python dependencies (from requirements.lock) ===")


@check("numpy 2",
       "uv pip sync requirements.lock --torch-backend cu129  (README: step 4)")
def _numpy():
    import numpy as np
    if int(np.__version__.split(".")[0]) < 2:
        raise RuntimeError(
            f"numpy {np.__version__}, but Sionna 2.x needs >= 2.2.6")
    return f"numpy {np.__version__}"


@check("torch",
       "uv pip sync requirements.lock --torch-backend cu129  (README: step 4)")
def _torch():
    import torch
    if torch.cuda.is_available():
        return f"torch {torch.__version__}  cuda: {torch.cuda.get_device_name(0)}"
    return f"torch {torch.__version__}  cuda: no (CPU only)"


@check("reinforcement learning and logging packages",
       "uv pip sync requirements.lock --torch-backend cu129  (README: step 4)")
def _rl():
    import gymnasium
    import matplotlib  # noqa: F401
    import minatar  # noqa: F401
    import pandas  # noqa: F401
    import psutil  # noqa: F401
    import tqdm  # noqa: F401
    import wandb  # noqa: F401
    return f"gymnasium {gymnasium.__version__}"


@check("GazeboEnv imports",
       "this needs both the workspace and the Python dependencies")
def _env():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo, "c_jepa", "control_jepa", "test"))
    import gazebo_env
    return gazebo_env.GazeboEnv.__name__


print("\n=== Sionna ===")


@check("backend selection",
       "sionna_compat.select_backend() must be called before importing "
       "sionna.rt  (README: Running on the GPU)")
def _backend():
    import sionna_compat
    variant = sionna_compat.select_backend(verbose=False)
    if variant.startswith("cuda"):
        return f"{variant}  (OptiX available, ray tracing on the GPU)"
    return (f"{variant}  (no OptiX: ray tracing falls back to the CPU. "
            "Install libnvidia-gl-<version> to use the GPU)")


if args.quick:
    print("  --  ray tracing and OFDM channel generation: skipped (--quick)")
else:
    @check("ray tracing and OFDM channel generation",
           "check the backend selection above first")
    def _rt():
        import sionna.rt as rt
        import sionna_compat
        from sionna.phy.channel import (cir_to_ofdm_channel,
                                        subcarrier_frequencies)
        from sionna.rt import (PathSolver, PlanarArray, Receiver, Transmitter,
                               load_scene)

        scene = load_scene(rt.scene.simple_street_canyon)
        array = PlanarArray(num_rows=1, num_cols=1, pattern="iso",
                            polarization="V")
        scene.tx_array = array
        scene.rx_array = array
        scene.add(Transmitter(name="tx", position=[-33, 0, 32]))
        scene.add(Receiver(name="rx", position=[20, 0, 1.7]))
        paths = PathSolver()(scene, max_depth=5)
        a, tau = sionna_compat.cir_for_ofdm(paths, normalize_delays=True)
        h = cir_to_ofdm_channel(subcarrier_frequencies(64, 30e3), a, tau,
                                normalize=False)
        arr = sionna_compat.to_numpy(h)
        return f"sionna {md.version('sionna')}  h {tuple(arr.shape)} {arr.dtype}"

failed = [label for label, ok in RESULTS if not ok]
print()
if failed:
    print(f"=== {len(failed)} of {len(RESULTS)} checks failed ===")
    for label in failed:
        print(f"  - {label}")
    sys.exit(1)

print(f"=== all {len(RESULTS)} checks passed ===")
print("A matplotlib warning about Axes3D above is expected and harmless; "
      "see Troubleshooting in README.md.")
sys.exit(0)
