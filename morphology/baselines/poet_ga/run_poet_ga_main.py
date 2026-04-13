import argparse
import os
import sys
import random
import time

import numpy as np
import torch

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..', '..')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(root_dir, 'externals', 'pytorch_a2c_ppo_acktr_gail'))

from run_poet_ga import run_poet_ga, evaluate_on_target
from ppo.arguments import get_args


def parse_poet_args():
    parser = argparse.ArgumentParser(description="POET+GA: multi-task GA with cross-task transfer")
    parser.add_argument("--tasks", nargs="+", required=True,
                        help="Training task names, e.g. Walker-v0 UpStepper-v0")
    parser.add_argument("--target-task", type=str, required=True,
                        help="Unseen target task for final evaluation")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Experiment directory name (auto-generated if omitted)")
    parser.add_argument("--pop-size", type=int, default=25,
                        help="Population size per task")
    parser.add_argument("--max-evaluations", type=int, default=1000,
                        help="Max evaluations per task before stopping")
    parser.add_argument("--train-iters", type=int, default=1000,
                        help="PPO training iterations per robot")
    parser.add_argument("--num-cores", type=int, default=25,
                        help="Parallel worker processes")
    parser.add_argument("--transfer-interval", type=int, default=5,
                        help="Attempt cross-task transfer every N generations")
    parser.add_argument("--transfer-num", type=int, default=2,
                        help="Number of elite morphologies to attempt transferring")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--num-processes", type=int, default=4,
                        help="Parallel environments per robot")
    parser.add_argument("--eval-interval", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":

    torch.multiprocessing.set_start_method("spawn")

    poet_args = parse_poet_args()

    random.seed(poet_args.seed)
    np.random.seed(poet_args.seed)

    args = get_args(argv=[])
    args.num_processes = poet_args.num_processes
    args.cuda = False
    args.no_cuda = True
    args.eval_interval = poet_args.eval_interval
    args.seed = poet_args.seed

    training_tasks = poet_args.tasks
    target_task = poet_args.target_task

    if poet_args.experiment_name is not None:
        exp_name = poet_args.experiment_name
    else:
        exp_name = "POET_GA_{}_{}".format(
            "_".join(t.split("-")[0].lower() for t in training_tasks),
            time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()),
        )

    task_states = run_poet_ga(
        args=args,
        tasks=training_tasks,
        structure_shape=(5, 5),
        pop_size=poet_args.pop_size,
        max_evaluations=poet_args.max_evaluations,
        train_iters=poet_args.train_iters,
        num_cores=poet_args.num_cores,
        experiment_name=exp_name,
        transfer_interval=poet_args.transfer_interval,
        transfer_num=poet_args.transfer_num,
    )

    evaluate_on_target(
        args=args,
        task_states=task_states,
        training_tasks=training_tasks,
        target_task=target_task,
        pop_size=poet_args.pop_size,
        train_iters=poet_args.train_iters,
        num_cores=poet_args.num_cores,
        experiment_name=exp_name,
    )
