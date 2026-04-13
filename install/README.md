# Environment Setup Guide

> **Important:** This project is built on **Evolution Gym 1.0**. The official EvoGym repository has since released version 2.0 with breaking API changes that are **not fully compatible** with our codebase. Please follow the steps below to install the correct version from the provided source archives.

## Prerequisites

- Python 3.7 or 3.8
- Linux, macOS, or Windows with [Visual Studio 2017](https://visualstudio.microsoft.com/vs/older-downloads/)
- [OpenGL](https://www.opengl.org/)
- [CMake](https://cmake.org/download/)

### Linux-Only Dependencies

```bash
sudo apt-get install xorg-dev libglu1-mesa-dev
```

## Step 1: Create Conda Environment

```bash
conda create --name evogym-robolda python=3.7.11
conda activate evogym-robolda
```

## Step 2: Extract Source Archives

Unzip the source archives provided in this directory:

```bash
unzip evogym-original.zip
unzip evogym-fix.zip
unzip GPyOpt-master.zip
unzip neat-python-master.zip
```

This will produce four directories: `evogym/`, `evogym-fix/`, `GPyOpt-master/`, and `neat-python-master/`.

## Step 3: Install Evolution Gym

```bash
cd evogym
pip install -r requirements.txt
python setup.py install
cd ..
```

## Step 4: Install Pyro (for RoboLDA)

```bash
pip install pyro-ppl
```

## Step 5: Install GPyOpt (for Bayesian Optimization baseline)

```bash
cd GPyOpt-master
python setup.py install
cd ..
```

## Step 6: Install NEAT-Python (for CPPN-NEAT baseline)

```bash
cd neat-python-master
python setup.py install
cd ..
```

## Verify Installation

Run the following test script to confirm everything is working:

```bash
cd install/evogym/examples
python gym_test.py
```

A window should open showing a random 5x5 robot taking random actions in the `Walker-v0` environment. Close the terminal process to exit.

## Alternative Setup for Universal Control Experiments

The universal control experiments require a modularized observation interface to match the input format of Transformers. We have modified the EvoGym source code to support this. To run these experiments, set up a **separate** conda environment using `evogym-fix` instead of the original `evogym`:

```bash
conda create --name evogym-universal python=3.7.11
conda activate evogym-universal
```

```bash
cd evogym-fix
pip install -r requirements.txt
python setup.py install
cd ..
```

Additionally, install the dependencies required for control baselines:

```bash
pip install scikit-learn IPython
```

Since universal control experiments do not involve morphology evolution, **GPyOpt and NEAT-Python are not required** in this environment.

## GPU Compatibility

The codebase was developed with PyTorch 1.10–1.12 (CUDA 10.2/11.3), which supports GPUs with compute capability sm_60–sm_80 (e.g., P100, V100, A100). If your GPU has a higher compute capability (e.g., A800 sm_80, H100 sm_90), you will need a newer PyTorch build:

```bash
# Example: PyTorch 1.13.1 with CUDA 11.6 (supports sm_37–sm_86)
pip install torch==1.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
```

Morphology experiments (training, generation, evaluation) run on CPU and do not require GPU. Control experiments use GPU by default; pass `--no-cuda` to any control script to run on CPU.

## Provided Source Archives

| Archive | Description |
|---|---|
| `evogym-original.zip` | Evolution Gym 1.0 (simulation environment for VSR) |
| `evogym-fix.zip` | Modified EvoGym with modularized observation interface (for universal control experiments) |
| `GPyOpt-master.zip` | GPyOpt (Bayesian Optimization library for BO baseline) |
| `neat-python-master.zip` | NEAT-Python (for CPPN-NEAT evolutionary baseline) |
