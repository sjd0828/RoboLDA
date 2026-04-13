# RoboLDA

Hierarchical probabilistic generative model for Voxel-based Soft Robot (VSR) design and control.
RoboLDA models the task-robot-organ-voxel hierarchy and enables zero-shot morphology generation for unseen tasks,
as well as organ-synergy-based universal control.

## Repository Structure

```
RoboLDA_repo_final/
├── install/                  Environment setup archives and instructions
├── survivors/                High-performing morphology samples (.npz)
├── demos/                    Qualitative results and visualizations
│   ├── control_qualitative/  Control experiment demo videos (Walker, Pusher)
│   └── hierarchical_structure/  Robot simulation GIFs for 8 tasks
├── morphology/               Morphology design experiments
│   ├── RoboLDA/              RoboLDA training, generation, conversion
│   ├── baselines/            Generative and evolutionary baselines
│   ├── ppo/                  PPO for single-robot morphology evaluation
│   ├── utils/                Shared utilities (Structure, mutation, mp)
│   └── evaluate.py           Batch morphology evaluation wrapper
└── control/                  Universal control experiments
    ├── run_metamorph_organ.py    MetaMorph + RoboLDA organ masks (ours)
    ├── run_metamorph.py          MetaMorph baseline
    ├── run_solar.py              SOLAR baseline (AP + two-level Transformer)
    ├── run_solar_ap.py           SOLAR-AP baseline (fixed AP masks)
    ├── ppo/                      PPO variants for universal control
    ├── utils/                    Shared utilities
    └── externals/                Transformer policy architectures
```

## 1. Environment Setup

See [`install/README.md`](install/README.md) for detailed instructions.

**Quick start:**

```bash
conda create -n robolda python=3.7.11 -y
conda activate robolda

# Install EvoGym (for morphology experiments)
cd install && unzip evogym-original.zip && cd evogym
pip install -r requirements.txt && python setup.py install && cd ..

# Install Pyro (for RoboLDA)
pip install pyro-ppl

# Install GPyOpt (for BO baseline)
unzip GPyOpt-master.zip && cd GPyOpt-master && python setup.py install && cd ..

# Install NEAT-Python (for CPPN-NEAT baseline)
unzip neat-python-master.zip && cd neat-python-master && python setup.py install && cd ..
```

For universal control experiments, set up a **separate** environment using `evogym-fix.zip`:

```bash
conda create -n robolda-control python=3.7.11 -y
conda activate robolda-control
cd install && unzip evogym-fix.zip && cd evogym-fix
pip install -r requirements.txt && python setup.py install && cd ..
pip install scikit-learn IPython
```

**GPU Compatibility Note:**
The codebase was developed and tested with PyTorch 1.10–1.12 (CUDA 10.2/11.3) on NVIDIA GPUs with compute capability sm_60–sm_80 (e.g., P100, V100, A100). If you are using newer GPUs (e.g., A800 sm_80, H100 sm_90), you may need to install a newer PyTorch version that supports your GPU's compute capability while maintaining compatibility with Python 3.7:

```bash
# Example for CUDA 11.6 (supports up to sm_86):
pip install torch==1.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
```

Morphology experiments run on CPU and do not require GPU. Control experiments benefit from GPU acceleration; pass `--no-cuda` to fall back to CPU if needed.

## 2. Morphology Design Experiments

### 2.1 RoboLDA Training

Train RoboLDA on survivor morphologies from specified tasks:

```bash
cd morphology/RoboLDA

python train.py \
    --survivors-root ../../survivors \
    --tasks walker pusher carrier \
    --save-dir results/walker_pusher_carrier \
    --epochs 300 \
    --individual-num 6 \
    --organ-num 6 \
    --seed 111
```

Output: `results/<name>/params/` (logtheta_loc.pt, organ_weights.pt), `train/` and `test/` directories with per-task morph.pt and organ.pt.

### 2.2 Zero-Shot Morphology Generation

Generate morphologies for an unseen target task. Only the organ-voxel layers (θ₃, θ₄) and the organ weight network are transferred from training. The task-individual layers (θ₁, θ₂) are sampled from their N(0,I) prior each time, ensuring the model generalizes without requiring a target task label:

```bash
python generate.py \
    --model-dir results/walker_pusher_carrier/params \
    --num-generate 25 \
    --save-dir generated/bridgewalker \
    --seed 0
```

### 2.3 Morphology Evaluation (PPO)

Evaluate generated morphologies on the target task:

```bash
cd ..
python evaluate.py \
    --structures-dir RoboLDA/generated/bridgewalker \
    --task BridgeWalker-v0 \
    --train-iters 1000 \
    --num-cores 4
```

### 2.4 Generative Baselines (MorphVAE, RoboGAN, c-DM, LASeR)

```bash
cd baselines

# Train + generate (preset 1: Walker+Pusher+Carrier -> BridgeWalker)
python -m gen_baselines --preset 1 --method morphvae --survivors_root ../../survivors --out_dir output --vae_epochs 300

# Run all four methods at once
python -m gen_baselines --preset 1 --method all --survivors_root ../../survivors --out_dir output

# Evaluate zero-shot generated morphologies
python -m gen_baselines.eval_zero_shot_generated --preset 1 --train_iters 1000
```

### 2.5 POET+GA

POET-style baseline: runs multi-task GA with periodic cross-task morphology transfer across training tasks, then evaluates the top `pop_size` (25) morphologies from the final generation on the unseen target task (split evenly across training tasks, PPO-trained from scratch):

```bash
cd baselines/poet_ga

python run_poet_ga_main.py \
    --tasks Walker-v0 UpStepper-v0 \
    --target-task DownStepper-v0 \
    --pop-size 25 \
    --max-evaluations 1000 \
    --train-iters 1000 \
    --num-cores 25 \
    --transfer-interval 5
```

### 2.6 Evolutionary Algorithm Baselines (GA / BO / CPPN-NEAT)

The full multi-generation evolutionary optimization code (GA, Bayesian Optimization, CPPN-NEAT) is provided by the EvoGym framework itself. After installing EvoGym (see Section 1), you can find the entry-point scripts and their supporting modules under the EvoGym source tree:

```
evogym/
└── examples/
    ├── run_ga.py            # GA entry point
    ├── run_bo.py            # Bayesian Optimization entry point
    ├── run_cppn_neat.py     # CPPN-NEAT entry point
    ├── ga/run.py            # GA core logic (tournament selection + mutation)
    ├── bo/run.py             # BO core logic (GPyOpt-based)
    ├── cppn_neat/run.py      # CPPN-NEAT core logic (neat-python)
    ├── ppo/                  # PPO controller training (shared by all EAs)
    └── utils/                # Shared utilities (Structure, mp_group, etc.)
```

**Running full evolutionary optimization** (example with GA):

```bash
cd <evogym-source>/examples

# Specify the target task via --env-name
python run_ga.py \
    --env-name Walker-v0 \
    --pop-size 25 \
    --max-evaluations 1000 \
    --train-iters 1000 \
    --num-cores 4
```

Replace `run_ga.py` with `run_bo.py` or `run_cppn_neat.py` for the other two algorithms. Results are saved to `examples/saved_data/<experiment_name>/`.

**Generation-0 baselines** (random morphologies from each EA's initializer, evaluated without evolution):

```bash
cd morphology/baselines
python -m gen_baselines.eval_ea_gen0_baselines --presets 1 --train_iters 1000
```

## 3. Universal Control Experiments

All control scripts use the same basic interface. First, train RoboLDA to obtain morphologies and organ assignments (Section 2.1).

### 3.1 MetaMorph + RoboLDA Organs (Ours)

```bash
cd control

python run_metamorph_organ.py \
    --task Walker-v0 \
    --morph-dir ../morphology/RoboLDA/results/walker_pusher_carrier/train/walker \
    --organ-dir ../morphology/RoboLDA/results/walker_pusher_carrier/train/walker \
    --seed 0 \
    --num-robots 10 \
    --train-iters 2000 \
    --num-cores 10
```

### 3.2 MetaMorph Baseline (No Organs)

```bash
python run_metamorph.py \
    --task Walker-v0 \
    --morph-dir ../morphology/RoboLDA/results/walker_pusher_carrier/train/walker \
    --seed 0 \
    --num-robots 10 \
    --train-iters 2000
```

### 3.3 SOLAR Baseline (AP Clustering + Two-Level Transformer)

```bash
python run_solar.py \
    --task Walker-v0 \
    --morph-dir ../morphology/RoboLDA/results/walker_pusher_carrier/train/walker \
    --seed 0 \
    --num-robots 10 \
    --train-iters 2000 \
    --ap-update-interval 100
```

### 3.4 SOLAR-AP Baseline (Fixed AP Masks)

```bash
python run_solar_ap.py \
    --task Walker-v0 \
    --morph-dir ../morphology/RoboLDA/results/walker_pusher_carrier/train/walker \
    --seed 0 \
    --num-robots 10 \
    --train-iters 2000
```

## 4. Task Combinations

The following 9 transfer presets are used in the paper:

| Preset | Pre-training Tasks | Target Task |
|--------|-------------------|-------------|
| 1 | Walker, Pusher, Carrier | BridgeWalker |
| 2 | BridgeWalker | Walker |
| 3 | Walker, Carrier | Pusher |
| 4 | Walker, Pusher | Carrier |
| 5 | Walker, UpStepper | DownStepper |
| 6 | Walker, DownStepper | UpStepper |
| 7 | Climber | Climber-v1 |
| 8 | Climber | Climber-v2 |
| 9 | PlatformJumper | GapJumper |

## 5. Survivor Data

The `survivors/` directory contains high-performing morphologies extracted from GA experiments.
Each `.npz` file stores a 5x5 voxel grid as `arr_0`.

| Directory | Task | Count |
|-----------|------|-------|
| walker | Walker-v0 | 230 |
| bridgewalker | BridgeWalker-v0 | 214 |
| carrier | Carrier-v0 | 187 |
| pusher | Pusher-v0 | 187 |
| upstepper | UpStepper-v0 | 187 |
| downstepper | DownStepper-v0 | 168 |
| climber | Climber-v0 | 187 |
| platformjumper | PlatformJumper-v0 | 171 |

## 6. Demos

### Qualitative Control Results

`demos/control_qualitative/` contains demo videos showing the learned organ-synergy control policy in action:

- `walker-demo.mp4` — Walker-v0 task
- `pusher-demo.mp4` — Pusher-v0 task

These correspond to the qualitative results in the control experiments section of the paper.

### Hierarchical Structure Visualizations

`demos/hierarchical_structure/` contains animated GIFs of 8 representative robots performing their respective tasks, illustrating the hierarchical task-robot-organ-voxel structure learned by RoboLDA. Each GIF shows the robot's body with a voxel-level grid overlay:

- `0_Walker-v0.gif`, `1_Carrier-v0.gif`, `2_Walker-v0.gif`, `3_DownStepper-v0.gif`
- `4_Walker-v0.gif`, `5_Pusher-v0.gif`, `6_Climber-v0.gif`, `7_BridgeWalker-v0.gif`
