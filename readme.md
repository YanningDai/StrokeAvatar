# StrokeAvatar

**StrokeAvatar** is a wearable-informed generative framework that reconstructs individualized post-stroke locomotor control from inertial sensing and predicts task-conditioned gait in unseen environments.

This repository contains code and data for the paper: *Wearable-informed generative digital avatars predict task-conditioned post-stroke locomotion*

## System Requirements

StrokeAvatar includes a Unity Editor for dataset visualization and a Python environment for training, which can be used independently. For example, the Unity Editor can be run on macOS for visualization, while training is performed on a Linux machine.

- **Tested platforms:** Windows 10/11, macOS (Apple Silicon M3), Linux Ubuntu 24.04.3 LTS.

- **Unity:** 2021.3 or later (Unity 6000.3 required on Ubuntu 24.04.3 LTS).  

- **Python:** Python 3.8.13 with ML-Agents 0.30.0.

## Installation Guide

Typical installation time: **~10 minutes** (excluding Unity Editor download)

### Unity Editor Setup

1. Install Unity Hub and Unity Editor (2021.3 or later) from  https://unity.com/download  

   > **Note:** Ubuntu 24.04 users should install Unity **6000.x** (Unity 6) instead of 2021.3 LTS.

2. Clone this repository:`git clone git@github.com:YanningDai/StrokeAvatar.git`

3. Open the project in the Unity Editor. The required packages will be automatically installed.

### Python Environment Setup

**Windows and Linux**

```bash
conda create -n strokeavatar python=3.8.13 -y
conda activate strokeavatar
python -m pip install mlagents==0.30.0
```
> **Windows note**: PyTorch may need to be installed manually before installing ML-Agents `pip3 install torch~=1.7.1 -f https://download.pytorch.org/whl/torch_stable.html`

**macOS (Apple Silicon)**

```bash
# Set up conda environment
git clone https://github.com/AmineAndam04/Conda-Env-Unity-ML-Agent.git
cd Conda-Env-Unity-ML-Agent
conda env create -f ml-agents.yaml -n strokeavatar
conda activate strokeavatar

# Install ML-Agents from source
git clone --branch release_20 https://github.com/Unity-Technologies/ml-agents.git
cd ml-agents
python -m pip install ./ml-agents-envs
python -m pip install ./ml-agents --no-deps
```


## Repository Structure

| Directory | Description |
|-----------|-------------|
| `Assets/StreamingAssets/` | Raw motion data and processed datasets |
| `Assets/ML-Agents/StrokeAvatar/Scenes/` | Unity environments: `Database`, `Controller`, `Predictor` |
| `Assets/ML-Agents/StrokeAvatar/Config/` | YAML configuration files |


## Environments

- **Database** — Process and visualize motion data. Press **Play** to start.
  - **Show Results** disabled: displays raw data (`StreamingAssets/NoitomData/`, `StreamingAssets/MocapData/`) and performs physics-based data processing.
  - **Show Results** enabled: displays processed datasets (`StreamingAssets/OutputHealthDataset/`, `StreamingAssets/OutputPatientDataset/`).
  - Select **ifHealth** to view healthy subjects; otherwise patient data is shown.

- **Controller** — Train locomotion controllers. Disable `usePatientData` for the healthy atlas model, or enable it with a patient data path for patient-specific training.

- **Predictor** — Predict patient locomotion under rehabilitation tasks (slope ascent, stair climbing). Toggle tasks via `Slope`/`Stair` objects in the scene. Supports curriculum learning with five difficulty levels.


## Training

Two training modes are supported:

### 1. Interactive Unity Environment

Used primarily for debugging and visualization within the Unity Editor.


```bash
mlagents-learn <config_file> --run-id <run_name>

# Example: 
mlagents-learn Assets/ML-Agents/StrokeAvatar/Config/Controller-health.yaml --run-id healthatlas_1
```

### 2. Standalone Built Environment

Recommended for large-scale training. First, build the Unity environment for your target operating system (macOS, Linux, or Windows), then train using the exported executable.:

```bash
mlagents-learn Assets/ML-Agents/StrokeAvatar/Config/Controller-health.yaml \
  --env env.app \
  --num-envs=20 \
  --run-id healthatlas_2
```
> Note: On Linux and Windows, replace env.app with the corresponding built executable.

Once training starts, real-time visualization is displayed automatically, and the terminal reports training logs (steps, rewards, and policy updates). Training curves can be monitored using **TensorBoard**. A full training run typically takes **~7 hours** on a single **NVIDIA RTX 3090 GPU**.

**Note**

- To continue training from an existing network (e.g., a trained healthy atlas),
  add `--initialize-from <run_id>` (e.g., `--initialize-from healthatlas_2`).

- If expert demonstrations are required (e.g., in the prediction environment),
  update the demonstration path in the YAML file to point to the corresponding
  recorded demo. Demonstrations can be recorded in the Controller environment
  using the Demonstration Recorder component.


## License

This project is released under the MIT License (OSI-approved).
The license permits free use, modification, and redistribution with attribution. See the `LICENSE` file for full details.