# DreamerV2 Meets Gazebo — C-JEPA & W-JEPA (ROS 2 Jazzy / Gazebo Harmonic port)

This repository provides the implementation of **Coupled Control-JEPA (C-JEPA)** and **Wireless-JEPA (W-JEPA)** for communication-aware remote robotic control.

The framework combines robot observations and dynamics from **ROS 2 / Gazebo** with wireless channel information generated using **Sionna RT**. C-JEPA is first pretrained in a Gym racing-car environment and then fine-tuned in Gazebo. W-JEPA is trained using CSI data collected from the synchronized Sionna RT environment.

This is a fork of [ICONgroupCWC/DreamerV2-Meets-Gazebo](https://github.com/ICONgroupCWC/DreamerV2-Meets-Gazebo), ported from ROS 1 Noetic + Gazebo Classic to **ROS 2 Jazzy + Gazebo Harmonic on Ubuntu 24.04**, and from Sionna RT 1.x to **Sionna 2.x**.

## What changed from upstream

| | Upstream | This fork |
| --- | --- | --- |
| OS | Ubuntu 20.04 | Ubuntu 24.04 LTS |
| ROS | Noetic (ROS 1) | Jazzy (ROS 2) |
| Gazebo | Gazebo Classic | Gazebo Harmonic (`gz-sim` 8) |
| ROS–Gazebo bridge | `gazebo_ros` | `ros_gz` 1.0.24 |
| Python | 3.8 | 3.12 |
| Sionna | RT 1.x API, TensorFlow backend | 2.x API, PyTorch backend |
| Python dependencies | installed ad hoc with `pip` | one locked virtual environment managed with [uv](https://docs.astral.sh/uv/) |
| `cv_bridge` | used for image conversion | replaced by `gz_sionna/src/ros_image.py` (see [The NumPy 2 problem](#the-numpy-2-problem)) |

---

## Requirements

Install these with `apt` before doing anything else. They are **not** managed by uv, because ROS 2 ships compiled C extensions that only work with the system Python interpreter.

| What | Package | Version here |
| --- | --- | --- |
| OS | — | Ubuntu 24.04.5 LTS |
| ROS 2 | `ros-jazzy-desktop` | Jazzy |
| Gazebo Harmonic | `ros-jazzy-ros-gz` | `ros_gz` 1.0.24, `gz-sim` 8.15.0 |
| Quaternion helpers | `ros-jazzy-tf-transformations` | 1.1.1 |
| URDF macros | `ros-jazzy-xacro` | 2.1.1 |
| Build tool | `python3-colcon-common-extensions` | 0.3.0 |
| Python package manager | [uv](https://docs.astral.sh/uv/) | 0.12 or newer |

```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-desktop \
  ros-jazzy-ros-gz \
  ros-jazzy-tf-transformations \
  ros-jazzy-xacro \
  python3-colcon-common-extensions
```

Gazebo Harmonic arrives through the `ros-jazzy-ros-gz` vendor packages, so there is no separate Gazebo installation step.

### For an NVIDIA GPU

Strongly recommended. Sionna RT ray tracing and DreamerV2 training both run far faster on a GPU.

Two separate things are needed, and having a working NVIDIA driver does **not** give you both:

| For | Library | Comes from |
| --- | --- | --- |
| PyTorch (training) | `libcuda.so.1` | `libnvidia-compute-<version>` |
| Sionna RT (ray tracing) | `libnvoptix.so.1` | `libnvidia-gl-<version>` |

Ubuntu splits the NVIDIA driver into a compute part and a graphics part, and **OptiX — the GPU ray tracing runtime Sionna RT needs — is in the graphics part**. A compute-only driver install gives you CUDA but no OptiX. Check and fix:

```bash
# is OptiX present?
ldconfig -p | grep libnvoptix || echo "OptiX missing"

# which driver version is loaded?
nvidia-smi --query-gpu=driver_version --format=csv,noheader

# install the matching graphics part (570 here -- use your own major version)
sudo apt install libnvidia-gl-570
```

Without it, ray tracing still works; it silently falls back to the CPU. See [Running on the GPU](#running-on-the-gpu).

If you do not have uv yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Set up the environment

Six steps. Run them in order; each one is explained below.

### 1. Create a colcon workspace and put the repository in it

ROS 2 builds a *workspace*, not a bare repository. The repository has to sit under `<workspace>/src/`.

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/Zerozuka/DreamerV2-Meets-Gazebo.git
```

If you already keep the clone somewhere else, symlink it instead of copying:

```bash
mkdir -p ~/ros2_ws/src
ln -s /path/to/your/DreamerV2-Meets-Gazebo ~/ros2_ws/src/DreamerV2-Meets-Gazebo
```

### 2. Make ROS 2 available in your shell

Every new terminal needs this. Nothing below works without it.

```bash
source /opt/ros/jazzy/setup.bash
```

Using zsh? Use `setup.zsh`, not `setup.bash`. Sourcing `setup.bash` from zsh breaks, because `$BASH_SOURCE` is empty there and the script looks for `setup.sh` in the current directory instead of in `/opt/ros/jazzy`.

```bash
source /opt/ros/jazzy/setup.zsh
```

### 3. Create the virtual environment

```bash
cd ~/ros2_ws/src/DreamerV2-Meets-Gazebo
uv venv --system-site-packages --python /usr/bin/python3.12 .venv
```

**The `--system-site-packages` flag is required.** Do not leave it out. `rclpy`, the ROS message packages and `tf_transformations` are installed by apt as compiled extensions built for `/usr/bin/python3.12`; pip cannot supply them. A normal, fully isolated virtual environment cannot see them, and `import rclpy` fails. This flag opens a window from the virtual environment onto the apt packages while still keeping everything uv installs separate.

`--python /usr/bin/python3.12` pins the interpreter to the apt one for the same reason: the compiled ROS 2 extensions are built against that exact interpreter, so a uv-downloaded Python of any version will not load them.

### 4. Install the Python dependencies

```bash
uv pip sync requirements.lock --torch-backend cu129
```

`requirements.lock` holds exact versions for all 104 packages, so every machine gets the same environment. It is generated from `requirements.in`, which is the file to edit when adding a dependency:

```bash
uv pip compile requirements.in -o requirements.lock --python-version 3.12 --torch-backend cu129
```

`--torch-backend` selects which PyTorch build to install, and the same value is needed on both commands, because the PyTorch index that serves the `+cu129` wheels is not recorded in the lock file.

| Your machine | Use |
| --- | --- |
| NVIDIA GPU, driver 525 or newer | `--torch-backend cu129` (the default here) |
| No NVIDIA GPU | `--torch-backend cpu` |
| Not sure | `--torch-backend auto` — uv reads the driver and picks for you |

`cu129` is the default because it is what `auto` resolves to on the development machine (driver 570 / CUDA 12.8), and pinning it explicitly keeps the lock file reproducible, which `auto` would not. Note that PyPI's plain `torch` is built for CUDA 13 and reports *"The NVIDIA driver on your system is too old"* on a CUDA 12.x driver, which is why an explicit backend matters.

If you change the backend, regenerate the lock with the same flag and re-run `uv pip sync`.

### 5. Build the workspace

```bash
cd ~/ros2_ws
colcon build --symlink-install
```

`--symlink-install` means edits to Python files and world files take effect without rebuilding. You still need to rebuild after adding a new file, or after changing a `package.xml`, `setup.py` or launch file.

### 6. Make the workspace available in your shell

```bash
source ~/ros2_ws/install/setup.bash
```

Same as step 2: every new terminal needs this, and zsh users want `setup.zsh`.

### Verify the setup

This checks everything at once and prints a line per component.

```bash
cd ~/ros2_ws/src/DreamerV2-Meets-Gazebo
.venv/bin/python - <<'PY'
import importlib.metadata as md, os, sys, numpy as np
import rclpy, tf_transformations, cv2
from nav_msgs.msg import Odometry
from ament_index_python.packages import get_package_share_directory as share
print("python", sys.version.split()[0], "numpy", np.__version__, "cv2", cv2.__version__)

sys.path.insert(0, os.path.join(share("gz_sionna"), "src"))
import sionna_compat
sionna_compat.select_backend()
from sionna.rt import load_scene, PathSolver, Transmitter, Receiver, PlanarArray
from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies
import sionna.rt as rt

scene = load_scene(rt.scene.simple_street_canyon)
scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
scene.add(Transmitter(name="tx", position=[-33, 0, 32]))
scene.add(Receiver(name="rx", position=[20, 0, 1.7]))
a, tau = sionna_compat.cir_for_ofdm(PathSolver()(scene, max_depth=5), normalize_delays=True)
h = cir_to_ofdm_channel(subcarrier_frequencies(64, 30e3), a, tau, normalize=False)
print("sionna", md.version("sionna"), "torch", md.version("torch"), "-> h", tuple(h.shape))

import torch
print("torch cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(CPU only)")
import mitsuba as mi
print("mitsuba variant:", mi.variant())
PY
```

Expected output on a machine with a GPU **and** OptiX installed:

```text
python 3.12.3 numpy 2.5.3 cv2 5.0.0
[sionna_compat] mitsuba variant = cuda_ad_mono_polarized
sionna 2.2.0 torch 2.13.0+cu129 -> h (1, 1, 1, 1, 1, 1, 64)
torch cuda: True NVIDIA GeForce RTX 3090
mitsuba variant: cuda_ad_mono_polarized
```

Without OptiX — that is, with `libnvidia-gl-<version>` missing — ray tracing still works but runs on the CPU, and you get two extra lines instead:

```text
python 3.12.3 numpy 2.5.3 cv2 5.0.0
[sionna_compat] cuda_ad_mono_polarized は使えない (RuntimeError)
[sionna_compat] mitsuba variant = llvm_ad_mono_polarized
sionna 2.2.0 torch 2.13.0+cu129 -> h (1, 1, 1, 1, 1, 1, 64)
torch cuda: True NVIDIA GeForce RTX 3090
mitsuba variant: llvm_ad_mono_polarized
```

PyTorch and Sionna RT are independent here: `torch cuda: True` together with `mitsuba variant: llvm_...` means training uses the GPU while ray tracing does not. See [Running on the GPU](#running-on-the-gpu).

---

## Running

Open two terminals. In **both** of them, first run the two `source` lines:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
```

### Terminal 1 — Gazebo

```bash
ros2 launch gz_sionna jetbot_tellus.launch.py
```

Without a GUI, for batch runs:

```bash
ros2 launch gz_sionna jetbot_tellus.launch.py gui:=false
```

The GUI costs almost nothing here, which is not the default behaviour of `gz sim` and is worth knowing about if you launch Gazebo by hand. Measured on the development machine (RTX 3090, driver 570), same world and same camera sensor, over a remote desktop:

| | Real-time factor | Camera | GPU |
| --- | --- | --- | --- |
| `gui:=false` | 0.96 | 28.7 fps | 33 %, 561 MiB |
| `gui:=true` | **0.98** | 28.0 fps | 36 %, 722 MiB |
| plain `gz sim -r <world>` with `DISPLAY` set | **0.068** | 2.5 fps | 0 %, 2 MiB |

The third row is what you get from a single `gz sim` process, and it is 14x slower. The cause is `DISPLAY`: when it is set, the **server** picks GLX rather than EGL for its camera sensor rendering, and on an X server without hardware GLX — a remote desktop such as NoMachine or VNC, or X forwarded over SSH — that falls back to software and drags the physics down with it.

The launch file avoids this by always starting the server as `gz sim -s -r --headless-rendering` and attaching `gz sim -g` as a **separate process** when `gui:=true`. `--headless-rendering` pins the server to EGL regardless of `DISPLAY`, so camera sensors stay on the GPU; the GUI window is still software-rendered over a remote desktop, but it is a different process and no longer holds the simulation back.

If you launch `gz sim` yourself, pass the same flags.

Useful launch arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `gui` | `true` | `false` runs the server only |
| `world` | `tellus3.world` | world file in `gz_sionna/worlds/` |
| `robot_name` | `jetbot_1` | robot namespace |
| `x_pos` `y_pos` `z_pos` `yaw` | on the track | spawn pose |
| `bridge_sensors` | `true` | `false` drops camera and IMU from the bridge |
| `view_result_image` | `false` | `true` also starts the image viewer |

Check that the robot is publishing:

```bash
ros2 topic list
ros2 topic hz /odom
ros2 topic hz /image_raw2
```

You should see `/odom`, `/image_raw2`, `/cmd_vel`, `/clock`, `/imu`, `/scan`, `/joint_states`. Drive the robot by hand to confirm it moves:

```bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
  '{linear: {x: 0.3}, angular: {z: 0.0}}'
```

### Terminal 2 — the Python side

Always call the interpreter inside `.venv`. Plain `python3` is the apt interpreter and does not have Sionna, PyTorch or Gymnasium.

```bash
cd ~/ros2_ws/src/DreamerV2-Meets-Gazebo
.venv/bin/python c_jepa/control_jepa/test/train_gazebo.py
```

`uv run` also works and picks up `.venv` automatically:

```bash
uv run c_jepa/control_jepa/test/train_gazebo.py
```

### The three training stages

**1. C-JEPA pre-training** in the Gym racing-car environment. No Gazebo needed.

```bash
cd ~/ros2_ws/src/DreamerV2-Meets-Gazebo/c_jepa/control_jepa/test
../../../.venv/bin/python train.py
```

**2. C-JEPA fine-tuning in Gazebo.** Start Gazebo in terminal 1 first, then:

```bash
cd ~/ros2_ws/src/DreamerV2-Meets-Gazebo
.venv/bin/python c_jepa/control_jepa/test/train_gazebo.py
```

**3. W-JEPA training** on CSI generated from Sionna RT, synchronized with robot pose, velocity and the C-JEPA latent control state.

The script the upstream README names for this stage, `c_jepa/wireless_jepa/src/train.py`, **does not exist in the repository**. See [Known gaps](#known-gaps). Channel generation itself runs:

```bash
.venv/bin/python c_jepa/control_jepa/test/channel_generate.py
```

### The coupled framework

Three terminals. Gazebo in the first, then:

```bash
# terminal 2 — C-JEPA
.venv/bin/python c_jepa/control_jepa/test/Gazebo_model_test.py
```

```bash
# terminal 3 — W-JEPA
.venv/bin/python c_jepa/wireless_jepa/src/wireless_jepa.py
```

The C-JEPA side needs trained weights, which are not included in the repository. The W-JEPA side does not, and runs on its own against a live Gazebo:

```text
Gazebo ──/odom───────────┐
                         ├──→ wireless_jepa.py (Sionna RT)
C-JEPA ──/render_trigger─┘         │
                                   ├──/channels (Float32MultiArray) ──→ C-JEPA
                                   └──/render_done (Int32) ───────────→ C-JEPA
```

`wireless_jepa.py` places the transmitter at the robot's `/odom` pose inside the Sionna RT scene, traces paths, converts the CIR to an OFDM channel and publishes it. `/channels` carries 768 floats laid out as `(2 real/imag, 3 receivers, 8, 16 subcarriers)`. Measured on the development machine: `/channels` at 6.2 Hz, and a `/render_trigger` → `/render_done` round trip of 0.09–0.13 s.

To check the coupling without trained weights, drive the robot and watch `/channels` change. Moving 1.68 m changed the channel by 1.8e-3, against a stationary ray-tracing jitter of 3.8e-7 — a factor of about 4600.

`gz_sionna/src/sionna_pos.py` is the visualisation counterpart. It follows `/odom` the same way but writes PNGs instead of publishing: `img/scene_N.png` renders the scene with the traced paths, `graph/scene_N.png` plots the channel impulse response. It takes relative output paths, so run it from a directory that already contains `img/`, `img2/` and `graph/`.

---

## Why there is only one environment

Sionna 2.x requires `numpy>=2.2.6`. ROS 2 Jazzy's apt packages were compiled against NumPy 1. That sounds like it forces two separate environments, and it nearly did. Measuring it showed otherwise.

### The NumPy 2 problem

Under NumPy 2 with `--system-site-packages`, only two apt packages break, and both have a pip replacement:

| Component | NumPy 2 | Fix |
| --- | --- | --- |
| `rclpy` | works | — |
| all message packages (`geometry_msgs`, `nav_msgs`, `sensor_msgs`, `std_msgs`, `rosgraph_msgs`) | works | — |
| `ament_index_python` | works | — |
| `tf_transformations` | **fails** | `transforms3d` from PyPI. The apt build calls `np.maximum_sctype`, removed in NumPy 2.0 |
| `cv2` | **fails** | `opencv-python` from PyPI. The apt build (4.6.0) is compiled against the NumPy 1 C API |
| `cv_bridge` image conversion | **fails, unfixable** | replaced, see below |

`cv_bridge` is the one that cannot be fixed with pip. Its `cv_bridge_boost.so` is a compiled C++ extension that only exists as an apt binary. Under NumPy 2 its initialization fails internally, leaving the conversion table empty, and any conversion dies with `KeyError: 16`.

Keeping `cv_bridge` would have meant splitting the workspace into a NumPy 1 environment and a NumPy 2 one. That split does not actually work here: `sionna_pos.py`, `wireless_jepa.py` and `channel_generate.py` each import **both** `rclpy` and `sionna` in the same process, so neither environment could run them.

Every one of the 55 `cv_bridge` call sites in this repository used the same single form:

```python
bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
```

So that one conversion is now implemented directly in `gz_sionna/src/ros_image.py` as `imgmsg_to_bgr8(msg)`, and `cv_bridge` is gone. Its output was checked against `cv_bridge` on NumPy 1 and matches exactly for `rgb8`, `bgr8`, `mono8`, `rgba8` and `bgra8`. Note that the conversion is not just a reshape: Gazebo's camera publishes `rgb8`, so the red and blue channels have to be swapped.

The one deliberate difference: `cv_bridge` ignores row padding and reads `step / channels` pixels per row, which shifts rows when `step > width * channels`. `ros_image.py` honours `step` and crops to `width`. Gazebo's `ros_gz` bridge publishes `step == width * 3`, so the difference never shows up in practice.

### Running on the GPU

PyTorch and Sionna RT each need a different piece of the NVIDIA stack, and they fail independently. Getting one working does not get you the other.

#### PyTorch

A CUDA build of PyTorch has to match the driver's CUDA version. PyPI's plain `torch` targets CUDA 13; on a CUDA 12.x driver it loads but refuses to use the GPU:

```text
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old
(found version 12080).
```

Pick the build with `--torch-backend`, as described in [step 4](#4-install-the-python-dependencies). Check the result with:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

#### Sionna RT and OptiX

Sionna RT 2.x pins the Mitsuba variant to `cuda_ad_mono_polarized` at import time. That variant does ray tracing through **OptiX**, which is a different library from CUDA:

| | Library | Purpose |
| --- | --- | --- |
| CUDA | `libcuda.so.1` | general GPU compute — what PyTorch uses |
| OptiX | `libnvoptix.so.1` | GPU ray tracing — what Sionna RT uses |

`libnvoptix.so.1` ships in `libnvidia-gl-<version>`, not in the compute-only driver package. A machine can therefore have a working GPU, a current driver and CUDA, and still have no OptiX. Without it, `load_scene()` fails:

```text
RuntimeError: [parser.cpp:1718] failed to instantiate scene plugin of
type "scene": Could not initialize OptiX!
```

Installing the graphics part of the driver fixes it:

```bash
sudo apt install libnvidia-gl-570   # match your driver major version
```

In a container, OptiX additionally needs the `graphics` driver capability. The default `NVIDIA_DRIVER_CAPABILITIES=compute,utility` does not include it, and the symptom is identical to a missing package:

```bash
docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics ...
```

#### The fallback

`gz_sionna/src/sionna_compat.py` makes the workspace run either way. Call `select_backend()` **before** importing `sionna.rt`:

```python
import sionna_compat
sionna_compat.select_backend()
import sionna.rt
```

It tries to build a minimal scene with each variant in turn — setting the variant alone does not raise, the error only surfaces when a scene is created — and falls back to `llvm_ad_mono_polarized` (CPU) when OptiX is unavailable. Set `SIONNA_MI_VARIANT` to force a specific variant.

---

## Repository layout

```text
gz_sionna/gz_sionna/
  launch/        jetbot_tellus.launch.py — spawns the world, the robot and the ros_gz bridge
  worlds/        tellus3.world, tellus3_with_road.world
  models/        Gazebo models (tellus arena, ball, cube, cylinder, radio_tower_)
  config/        cross_markers_400.csv, path_points.csv — the track data GazeboEnv reads
  src/           modules shared by every package:
                   ros_image.py      sensor_msgs/Image -> bgr8, replaces cv_bridge
                   sionna_compat.py  Sionna 2.x backend selection and CIR shape fixes
                   ros1_compat.py    ROS 1 rospy API reimplemented on rclpy
                   gz_world_control.py  gz-transport replacement for gazebo_msgs services
                   paths.py          model and output directory resolution
                   sionna_pos.py     Sionna RT node tracking the robot pose
                   wireless_jepa.py  W-JEPA node

jetbot_world/    Jetbot URDF/xacro and the Gazebo plugin configuration
img_showing/     camera image viewer node
c_jepa/
  control_jepa/test/   C-JEPA: gazebo_env.py (the Gymnasium environment), training
                       and evaluation scripts, and a bundled copy of dreamerv2/
  wireless_jepa/src/   W-JEPA: wireless_jepa.py and another copy of dreamerv2/
```

`requirements.in` and `requirements.lock` are at the repository root, next to `.venv`.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'rclpy'`** — you either forgot `source /opt/ros/jazzy/setup.bash`, or the virtual environment was created without `--system-site-packages`. Delete `.venv` and redo step 3.

**`ModuleNotFoundError: No module named 'sionna'` / `'torch'` / `'gymnasium'`** — you are running the apt interpreter. Use `.venv/bin/python` or `uv run`.

**`ModuleNotFoundError: No module named 'paths'` / `'ros_image'` / `'gz_world_control'`** — the workspace is not sourced, or not built. Run `colcon build --symlink-install` in `~/ros2_ws`, then `source ~/ros2_ws/install/setup.bash`. These modules are resolved through `ament_index` from `gz_sionna`'s share directory.

**Sourcing ROS 2 breaks in zsh** — use `setup.zsh` instead of `setup.bash`, in both places.

**`gz: command not found`** — the real binary is `/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz`, and it is only on `PATH` after sourcing ROS 2.

**The GUI never appears, and the log says `could not connect to display`** — you are in a terminal that has no `DISPLAY`, such as an SSH login or a plain console. Point it at the desktop session you want the window to open in:

```bash
ls /tmp/.X11-unix/        # X1001 means display :1001
export DISPLAY=:1001
```

`DISPLAY` only decides which X server the window goes to; it does not decide where OpenGL runs, so it has no effect on the real-time factor. The display number is assigned per remote-desktop session and changes when you reconnect, so check it rather than hard-coding it.

**Gazebo shows a black window, or crashes on start, over NoMachine / VNC / any virtual display** — set the Qt platform, and force software rendering only if that is not enough:

```bash
export QT_QPA_PLATFORM=xcb
export LIBGL_ALWAYS_SOFTWARE=1   # only if the window is still black or Gazebo crashes
```

With `libnvidia-gl-<version>` installed, the GUI starts over NoMachine without `LIBGL_ALWAYS_SOFTWARE` at all, so try `QT_QPA_PLATFORM=xcb` on its own first.

**The simulation crawls and the robot barely moves** — check the real-time factor, shown in the bottom-left of the GUI or on the stats topic:

```bash
gz topic -e -t /stats -n 1 | grep real_time_factor
```

Around 0.07 means the server is rendering its camera sensors in software. That happens when a single `gz sim` process runs both the server and the GUI with `DISPLAY` set; see [Terminal 1 — Gazebo](#terminal-1--gazebo). The launch file already avoids it, so you only hit this by starting `gz sim` by hand — add `-s --headless-rendering` and run `gz sim -g` separately. The robot is not broken: it is moving correctly in simulated time, just 14x slower in wall-clock time. The value printed in the first second or two after startup is meaningless, so let it settle.

**Stale Gazebo processes after a crash** — a second server on the same partition makes the simulation behave strangely. List them first, then stop them:

```bash
pgrep -af 'g[z] sim'
pkill -f 'g[z] sim'
```

Write the pattern as `g[z] sim`, not `gz sim`. The latter matches the `pkill` command line itself, so `pkill` kills its own shell.

**`GazeboEnv` hangs for 30 seconds and then raises** — no `/odom` or `/image_raw2` is arriving. Gazebo is not running, or `robot_name` does not match the namespace `GazeboEnv` subscribes to. Check with `ros2 topic hz /odom`.

**`FileNotFoundError` for `cross_markers_400.csv` or `path_points.csv`** — both live in `gz_sionna/config/` and are found through `ament_index`, so the workspace has to be built and sourced.

**`KeyError: 16` from `cv_bridge`** — something still imports `cv_bridge`. Use `imgmsg_to_bgr8` from `ros_image` instead; see [The NumPy 2 problem](#the-numpy-2-problem).

**`Could not initialize OptiX!`** — call `sionna_compat.select_backend()` before importing `sionna.rt`; see [Running on the GPU](#running-on-the-gpu).

---

## Known gaps

These are tracked as issues and are not regressions from the port.

- **Missing Gazebo models.** `tellus3_with_road.world` references models that upstream did not publish. `road_model`, `radio_tower_`, `cube`, `ball` and `cylinder` have been reconstructed; `cross_line`, `race_end` and `receiver_1..3` are still missing. `tellus3.world` opens fine and is the default.
- **`c_jepa/wireless_jepa/src/train.py` does not exist**, although the upstream README names it as the W-JEPA training entry point.
- **No trained weights** are included, so the evaluation scripts and the coupled demo cannot be run end to end from a fresh clone.
- **`control_jepa/test/pomdp.py` imports the pre-Gymnasium `gym` package**, which has no wheel for Python 3.12. It is therefore not installed. Everything else uses `gymnasium`.
- **`control_jepa/test/DQN_model_.py` needs `torchrl` and `tensordict`**, which are not installed either. It is an unused variant of `DQN_model.py`.
- **`gz_sionna/src/channel generation.py` is dead code** using the Sionna 1.x `scene.compute_paths()` API. It is superseded by `control_jepa/test/channel_generate.py`.
- **`dreamerv2/` exists twice**, byte-identical, under `control_jepa/test/` and `wireless_jepa/src/`. Because of that name collision the ROS packages do not install Python modules yet, which is why shared modules are reached through `ament_index` and `sys.path`.
- **Licensing is unresolved.** Every `package.xml` still says `<license>TODO</license>` and there is no LICENSE file.

---

## Simulation environments

The framework uses two synchronized simulation environments: **Gazebo** for robot simulation and **Sionna RT** for wireless channel simulation. Both represent the **same physical environment with identical geometry and spatial configuration**.

### Gazebo — robot environment

Gazebo provides robot dynamics, camera observations, robot states and control interfaces through ROS 2.

![Gazebo environment](images/gazebo_environment.jpg)

### Sionna RT — wireless environment

The same environment is reconstructed in Sionna RT, preserving geometry and coordinate system, and used for physics-based ray tracing, wireless channel modeling and CSI generation.

![Sionna RT environment](images/sionna_environment.png)

Robot position, orientation and motion are synchronized between Gazebo and Sionna RT, so the wireless channel is evaluated according to the robot's movement in Gazebo.

The Tellus arena model is taken from [ICONgroupCWC/Gazebo-Sionna-RT-Integration](https://github.com/ICONgroupCWC/Gazebo-Sionna-RT-Integration) (MIT).

## Demo in action

[![Coupled C-JEPA and W-JEPA Remote Robotic Control](https://img.youtube.com/vi/hw_bdS3P6Oc/0.jpg)](https://www.youtube.com/watch?v=hw_bdS3P6Oc)

## Contributors

Original work:

1. H.P. Madushanka ([madushanka.hewapathiranage@oulu.fi](mailto:madushanka.hewapathiranage@oulu.fi))
2. Sumudu Samarakoon ([sumudu.samarakoon@oulu.fi](mailto:sumudu.samarakoon@oulu.fi))
3. Mehdi Bennis ([mehdi.bennis@oulu.fi](mailto:mehdi.bennis@oulu.fi))

ROS 2 Jazzy / Gazebo Harmonic / Sionna 2.x port: Rei Ishizuka.
