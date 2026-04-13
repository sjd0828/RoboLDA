"""Evaluate generated morphologies on a target task using PPO."""

import argparse
import os
import sys
import random
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ppo.run_ppo import run_ppo
from ppo.arguments import get_args as get_ppo_args
from evogym.utils import get_full_connectivity
import utils.mp_group as mp
from utils.algo_utils import TerminationCondition, Structure


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate morphologies with PPO")
    parser.add_argument("--structures-dir", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--train-iters", type=int, default=1000)
    parser.add_argument("--num-cores", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.multiprocessing.set_start_method("spawn")
    random.seed(args.seed)
    np.random.seed(args.seed)

    ppo_args = get_ppo_args(argv=[])
    ppo_args.num_processes = 1
    ppo_args.cuda = False
    ppo_args.no_cuda = True
    ppo_args.eval_interval = 50
    ppo_args.seed = args.seed

    tc = TerminationCondition(args.train_iters)

    experiment_name = f"eval_{args.task}_{time.strftime('%Y-%m-%d-%H_%M_%S')}"
    save_dir = os.path.join("saved_data", experiment_name)
    os.makedirs(os.path.join(save_dir, "controller"), exist_ok=True)

    structures = []
    files = sorted(f for f in os.listdir(args.structures_dir) if f.endswith(".npz"))
    for i, fn in enumerate(files):
        data = np.load(os.path.join(args.structures_dir, fn))
        body = data["arr_0"]
        conn = get_full_connectivity(body)
        structures.append(Structure(body, conn, i, task_id=0))

    print(f"Loaded {len(structures)} structures")

    group = mp.Group()
    for structure in structures:
        ppo_run_args = (
            structure.label % 2, ppo_args, args.task, structure,
            tc, (os.path.join(save_dir, "controller"), structure.label)
        )
        group.add_job(run_ppo, ppo_run_args, callback=structure.set_reward)

    group.run_jobs(args.num_cores)

    for s in structures:
        s.compute_fitness()
    structures.sort(key=lambda s: s.fitness, reverse=True)

    print(f"\nResults for {args.task}:")
    for s in structures:
        print(f"  Robot {s.label}: fitness = {s.fitness:.2f}")

    mean_fit = np.mean([s.fitness for s in structures])
    max_fit = max(s.fitness for s in structures)
    print(f"\nMean fitness: {mean_fit:.2f}, Max fitness: {max_fit:.2f}")


if __name__ == "__main__":
    main()
