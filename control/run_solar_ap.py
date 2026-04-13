"""SOLAR-AP baseline: fixed AP-derived organ masks + MetaMorph backbone.

Computes Affinity Propagation clustering ONCE at initialization using
position-based affinity, then uses the resulting synergy masks as fixed
organ assignments throughout training. Uses metamorph_organ.ATTBase
(not the two-level model_lowrank.ATTBase) with run_ppo_universal.
"""

import os
import sys
import argparse
import random
import time

import numpy as np
import torch
from sklearn.cluster import AffinityPropagation

curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curr_dir)
sys.path.insert(1, os.path.join(curr_dir, "externals", "pytorch_a2c_ppo_acktr_gail"))

from ppo.run_ppo_universal import run_ppo_att
from ppo.arguments import get_args
from evogym.utils import get_full_connectivity

import utils.mp_group as mp
from utils.algo_utils import TerminationCondition, Structure

from a2c_ppo_acktr.metamorph_organ import ATTBase

HEAD_SIZES = {
    "Walker-v0": 2, "Carrier-v0": 6, "Pusher-v0": 6,
    "Climber-v0": 2, "UpStepper-v0": 14, "DownStepper-v0": 14,
    "BridgeWalker-v0": 3, "Catcher-v0": 7, "Balancer-v0": 1, "Thrower-v0": 6,
}


def parse_control_args():
    parser = argparse.ArgumentParser(
        description="SOLAR-AP baseline: fixed AP organ masks + MetaMorph backbone")
    parser.add_argument("--task", type=str, default="Walker-v0",
                        help="EvoGym task name")
    parser.add_argument("--morph-dir", type=str, required=True,
                        help="Directory containing morph.pt")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed")
    parser.add_argument("--num-robots", type=int, default=10,
                        help="Number of robots to sample")
    parser.add_argument("--train-iters", type=int, default=2000,
                        help="PPO training iterations per robot")
    parser.add_argument("--num-cores", type=int, default=10,
                        help="Number of parallel worker processes")
    parser.add_argument("--h-dim", type=int, default=None,
                        help="Task observation head size (default: from HEAD_SIZES)")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dimension for Transformer")
    parser.add_argument("--obs-dim", type=int, default=8,
                        help="Per-voxel observation dimension")
    parser.add_argument("--action-dim", type=int, default=1,
                        help="Per-voxel action dimension")
    parser.add_argument("--n-head", type=int, default=1,
                        help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate in Transformer")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--num-processes", type=int, default=1,
                        help="Number of parallel environments per robot")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="Rollout length per PPO update")
    parser.add_argument("--width", type=int, default=5,
                        help="Robot body grid width")
    parser.add_argument("--eval-interval", type=int, default=25,
                        help="Evaluate every N PPO updates")
    parser.add_argument("--log-interval", type=int, default=5,
                        help="Log every N PPO updates")
    parser.add_argument("--ap-affinity-scale", type=float, default=5.0,
                        help="Scale factor for position-based affinity")
    parser.add_argument("--ap-affinity-decay", type=float, default=0.1,
                        help="Decay rate for distance in affinity matrix")
    parser.add_argument("--ap-damping", type=float, default=0.5,
                        help="Damping factor for AP convergence")
    parser.add_argument("--output-name", type=str, default=None,
                        help="Experiment output directory name")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="Disable CUDA")
    return parser.parse_args()


def build_position_affinity(grid_size=5, decay=0.1, scale=5.0):
    """Build a position-based affinity matrix for a grid_size x grid_size body."""
    n = grid_size * grid_size
    affinity = np.zeros((n, n))
    for x in range(n):
        ix, jx = x // grid_size, x % grid_size
        for y in range(x + 1, n):
            iy, jy = y // grid_size, y % grid_size
            dist = abs(ix - iy) + abs(jx - jy)
            val = np.exp(-decay * dist) * scale
            affinity[x, y] = val
            affinity[y, x] = val
    return affinity


def compute_ap_organs(affinity, seed=0, damping=0.5):
    """Run AP clustering once and return one-hot organ masks (num_clusters, 25)."""
    ap = AffinityPropagation(
        affinity="precomputed", random_state=seed, damping=damping, verbose=False,
    ).fit(affinity)
    labels = ap.labels_
    num_clusters = int(np.max(labels)) + 1
    organs = np.zeros((num_clusters, 25))
    for c in range(num_clusters):
        organs[c, labels == c] = 1
    return torch.tensor(organs, dtype=torch.float32)


def run_universal(structures, organs_list, ppo_args, tc, save_dir, output_dir, temp_path):
    save_path_controller = os.path.join(save_dir, "controller")

    head_size = ppo_args.h_dim
    transformer_mu = ATTBase(
        h_dim=head_size, obs_dim=ppo_args.obs_dim,
        action_dim=ppo_args.action_dim, hidden_dim=ppo_args.hidden_dim,
        body_size=ppo_args.width, n_head=ppo_args.n_head,
        dropout=ppo_args.dropout,
    )
    transformer_v = ATTBase(
        h_dim=head_size, obs_dim=ppo_args.obs_dim,
        action_dim=1, hidden_dim=ppo_args.hidden_dim,
        body_size=ppo_args.width, n_head=ppo_args.n_head,
        dropout=ppo_args.dropout,
    )

    group = mp.Group()
    for idx, (structure, organ) in enumerate(zip(structures, organs_list)):
        job_args = (
            idx, ppo_args, ppo_args.task, structure, organ, tc,
            (save_path_controller, structure.label),
            transformer_mu, transformer_v, output_dir,
        )
        group.add_job(run_ppo_att, job_args, callback=structure.set_reward)

    group.run_jobs(ppo_args.num_cores)

    for structure in structures:
        structure.compute_fitness()

    with open(temp_path, "a") as f:
        for structure in structures:
            f.write("{}\t\t{}\n".format(structure.label, structure.fitness))


if __name__ == "__main__":
    ctrl_args = parse_control_args()

    task_short = ctrl_args.task.split("-")[0].lower()
    if ctrl_args.output_name is None:
        timestamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        exp_name = "{}-solar-ap-{}".format(task_short, timestamp)
    else:
        exp_name = ctrl_args.output_name

    torch.multiprocessing.set_start_method("spawn")
    random.seed(ctrl_args.seed)
    np.random.seed(ctrl_args.seed)

    ppo_args = get_args(argv=[])
    ppo_args.train_iters = ctrl_args.train_iters
    ppo_args.lr = ctrl_args.lr
    ppo_args.num_processes = ctrl_args.num_processes
    ppo_args.num_steps = ctrl_args.num_steps
    ppo_args.num_cores = ctrl_args.num_cores
    ppo_args.width = ctrl_args.width
    ppo_args.eval_interval = ctrl_args.eval_interval
    ppo_args.log_interval = ctrl_args.log_interval
    ppo_args.cuda = not ctrl_args.no_cuda
    ppo_args.no_cuda = ctrl_args.no_cuda
    ppo_args.action_dim = ctrl_args.action_dim
    ppo_args.obs_dim = ctrl_args.obs_dim
    ppo_args.hidden_dim = ctrl_args.hidden_dim
    ppo_args.n_head = ctrl_args.n_head
    ppo_args.dropout = ctrl_args.dropout
    ppo_args.seed = ctrl_args.seed
    ppo_args.task = ctrl_args.task

    if ctrl_args.h_dim is not None:
        ppo_args.h_dim = ctrl_args.h_dim
    else:
        ppo_args.h_dim = HEAD_SIZES[ctrl_args.task]

    tc = TerminationCondition(ppo_args.train_iters)

    save_dir = os.path.join(curr_dir, "saved_data", exp_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "controller"), exist_ok=True)
    temp_path = os.path.join(save_dir, "output.txt")
    output_dir = os.path.join(save_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    robos = torch.load(os.path.join(ctrl_args.morph_dir, "morph.pt"))
    indices = np.random.choice(robos.shape[0], size=ctrl_args.num_robots, replace=False)

    structures = []
    for label in indices:
        body = np.array(robos[label])
        conn = get_full_connectivity(body)
        structures.append(Structure(body, conn, label, task_id=0))
    print("Structures loaded: {} robots.".format(len(structures)))

    affinity = build_position_affinity(
        grid_size=ctrl_args.width,
        decay=ctrl_args.ap_affinity_decay,
        scale=ctrl_args.ap_affinity_scale,
    )
    ap_organs = compute_ap_organs(
        affinity, seed=ctrl_args.seed, damping=ctrl_args.ap_damping,
    )
    print("AP clustering: {} organ groups computed.".format(ap_organs.shape[0]))

    organs_list = [ap_organs.clone() for _ in range(ctrl_args.num_robots)]
    torch.save(ap_organs, os.path.join(save_dir, "ap_organs.pt"))

    run_universal(structures, organs_list, ppo_args, tc, save_dir, output_dir, temp_path)
    print("SOLAR-AP experiment finished.")
