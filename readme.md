# StrokeAvatar

Code and resources for the manuscript:

**Wearable-informed generative digital avatars predict task-conditioned post-stroke locomotion**


## Overview

**StrokeAvatar** is a Unity ML-Agents–based framework for learning wearable-informed generative digital avatars that predict task-conditioned locomotion in post-stroke patients.

The system integrates motion capture data, physics-based optimization, and reinforcement learning to model healthy locomotion priors and patient-specific adaptations across rehabilitation tasks.

## Requirements

- **Unity** ≥ 2021.3  
- **ML-Agents** ≥ Release 19  
- **Operating systems:** macOS, Linux, and Windows  
- Tested on Unity **2021.3** and **2022.3**, ML-Agents Release **19** and **20**

ML-Agents installation guide:  
https://github.com/Unity-Technologies/ml-agents/blob/release_20_docs/docs/Installation.md

## Python Environment

```bash
conda create -n strokeavatar python=3.8.13 -y
conda activate strokeavatar
python -m pip install mlagents==0.30.0
```
> Installing mlagents==0.30.0 will automatically install a compatible version of PyTorch and other required dependencies.
> 


## Repository Structure

- **Data**  
  `Assets/StreamingAssets/`  
  Contains both raw motion data and processed datasets.

- **Environments**  
  `Assets/ML-Agents/StrokeAvatar/Scenes/`  
  Three Unity environments corresponding to different stages of the pipeline:
  - `Database`
  - `Controller`
  - `Predictor`

- **Configurations**  
  `Assets/ML-Agents/StrokeAvatar/Config/`  
  YAML configuration files for different models and rehabilitation tasks.

## Environments and Visualization

Open the project directly in Unity for visualization and environment inspection.

### 1. Database Environment

- Press **Play** to process and visualize motion data.
- When **Show Results** is disabled:
  - Displays raw motion data from:
    - `Assets/StreamingAssets/NoitomData`
    - `Assets/StreamingAssets/MocapData`
  - Performs physics-based optimization and saves processed results.
- When **Show Results** is enabled:
  - Displays processed datasets stored in:
    - `Assets/StreamingAssets/OutputHealthDataset`
    - `Assets/StreamingAssets/OutputPatientDataset`
- In Play Mode, selecting **Health** visualizes healthy subjects; otherwise, patient data are shown.

### 2. Controller Environment

- **Healthy Atlas Model**  
  - Disable `usePatientData`
  - Trains an average healthy locomotion controller with randomized step length and speed.

- **Patient-Specific Controller**  
  - Enable `usePatientData`
  - Specify the path to an individual patient’s data to train a personalized control policy.

### 3. Prediction Environment

- Predicts patient locomotion under different rehabilitation tasks.
- Currently supported tasks:
  - Slope Ascent 
  - Stair Climbing
- Tasks are toggled by enabling/disabling `Slope` or `Stair` objects in the scene.
- Curriculum learning is supported via configuration files, with five difficulty levels from easy to hard.

## Training

Two training modes are supported on **macOS, Linux, and Windows**.

### 1. Interactive Unity Environment

Used primarily for debugging and visualization within the Unity Editor.


```bash
mlagents-learn <config_file> --run-id <run_name>

# Example: healthy atlas training
mlagents-learn Assets/ML-Agents/StrokeAvatar/Config/Controller-health.yaml --run-id healthatlas_1
```

### 2. Standalone Built Environment

Recommended for large-scale training.

First, build the Unity environment for your target operating system (macOS, Linux, or Windows), then train using the exported executable.:

```bash
mlagents-learn Assets/ML-Agents/StrokeAvatar/Config/Controller-health.yaml \
  --env env.app \
  --num-envs=20 \
  --run-id healthatlas_2
```
> Note: On Linux and Windows, replace env.app with the corresponding built executable.


Patient-specific controllers are initialized from the trained healthy atlas model
using `--initialize-from`:
```bash
mlagents-learn Assets/ML-Agents/StrokeAvatar/Config/Controller-patient.yaml \
  --env env.app \
  --run-id patient_01 \
  --num-envs=20 \
  --initialize-from healthatlas_2

```

For the prediction environment, the demonstration path specified in the YAML file must be updated to point to a patient-specific controller demonstration. Demonstrations are recorded in the Controller environment using the
**Demonstration Recorder** component. (see tutorials in https://www.youtube.com/watch?v=Dhr4tHY3joE)

## License

This repository is released for research purposes only.
Please refer to the manuscript for details on data usage and ethical approval.