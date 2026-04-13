"""Train RoboLDA on survivor morphologies via Stochastic Variational Inference."""

import argparse
import os
import random

import numpy as np
import pyro
import torch
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam
from tqdm import trange
from itertools import accumulate

from VAE_RoboLDA import RoboLDA
from vec2morph import morph_to_vec, vec_to_morph, operator


def parse_args():
    parser = argparse.ArgumentParser(description="Train RoboLDA on survivor morphologies")
    parser.add_argument("--survivors-root", type=str, required=True,
                        help="Path to survivors/ directory")
    parser.add_argument("--tasks", nargs="+", required=True,
                        help="Task subdirectory names (e.g. walker pusher carrier)")
    parser.add_argument("--save-dir", type=str, default="results",
                        help="Directory to save trained parameters")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--individual-num", type=int, default=6)
    parser.add_argument("--organ-num", type=int, default=6)
    parser.add_argument("--comp-num", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=111)
    return parser.parse_args()


def load_survivors(survivors_root, task_dirs):
    """Load .npz survivors and return one-hot encoded bodies and task labels."""
    all_bodies = []
    all_tasks = []
    file_counts = []
    task_num = len(task_dirs)

    for t_idx, task_dir in enumerate(task_dirs):
        path = os.path.join(survivors_root, task_dir)
        files = sorted(f for f in os.listdir(path) if f.endswith(".npz"))
        file_counts.append(len(files))
        task_oh = [0] * task_num
        task_oh[t_idx] = 1
        for fn in files:
            data = np.load(os.path.join(path, fn))["arr_0"]
            vec = morph_to_vec(data).reshape(1, 25 * 5)
            all_bodies.append(vec)
            all_tasks.append(torch.tensor(task_oh).reshape(1, -1))

    robos = torch.cat(all_bodies, dim=0)
    tasks = torch.cat(all_tasks, dim=0)
    return robos, tasks, file_counts


def train_test_split(file_counts, train_ratio, rng):
    """Return per-task lists of train indices (into the per-task file list)."""
    train_indices = []
    train_counts = []
    for count in file_counts:
        n_train = round(count * train_ratio)
        indices = rng.sample(range(count), n_train)
        train_indices.append(set(indices))
        train_counts.append(n_train)
    return train_indices, train_counts


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    task_dirs = args.tasks
    task_num = len(task_dirs)
    rng = random.Random(args.seed)

    # Load all survivors and get file counts per task
    all_bodies_list = []
    all_tasks_list = []
    file_counts = []
    per_task_files = []

    for t_idx, task_dir in enumerate(task_dirs):
        path = os.path.join(args.survivors_root, task_dir)
        files = sorted(f for f in os.listdir(path) if f.endswith(".npz"))
        file_counts.append(len(files))
        per_task_files.append(files)

    train_indices, train_counts = train_test_split(file_counts, args.train_ratio, rng)

    # Build train set
    train_robos_list = []
    train_tasks_list = []
    for t_idx, task_dir in enumerate(task_dirs):
        path = os.path.join(args.survivors_root, task_dir)
        task_oh = [0] * task_num
        task_oh[t_idx] = 1
        for j, fn in enumerate(per_task_files[t_idx]):
            if j in train_indices[t_idx]:
                data = np.load(os.path.join(path, fn))["arr_0"]
                train_robos_list.append(morph_to_vec(data).reshape(1, 25 * 5))
                train_tasks_list.append(torch.tensor(task_oh).reshape(1, -1))

    robos = torch.cat(train_robos_list, dim=0)
    tasks = torch.cat(train_tasks_list, dim=0)
    print(f"Training set: {robos.shape[0]} morphologies across {task_num} tasks")

    # Build model
    model = RoboLDA(
        task_num=task_num,
        individual_num=args.individual_num,
        organ_num=args.organ_num,
        comp_num=args.comp_num,
        voxel_num=5,
        robot_num=robos.shape[0],
        hidden=args.hidden,
        dropout=0,
    )

    ELBO = TraceEnum_ELBO(max_plate_nesting=1, strict_enumeration_warning=False)
    guide = config_enumerate(model.guide, expand=True)
    optim = Adam({"lr": args.lr})
    svi = SVI(model.model, guide, optim, loss=ELBO)

    # Training loop
    losses = []
    bar = trange(args.epochs)
    for epoch in bar:
        loss = svi.step(robos.float(), tasks.float())
        losses.append(loss)
        bar.set_postfix(epoch_loss="{:.2e}".format(loss))

    # Inference on train set
    (logtheta1_loc, logtheta2_loc, logtheta3_loc, logtheta3_scale,
     logtheta4_loc, logtheta4_scale, inds, organs,
     organ_logits) = model.guide(robos.float(), tasks.float())

    params_all = model.state_dict()

    # Save parameters
    params_dir = os.path.join(args.save_dir, "params")
    os.makedirs(params_dir, exist_ok=True)

    torch.save({
        "logtheta1_loc": logtheta1_loc,
        "logtheta2_loc": logtheta2_loc,
        "logtheta3_loc": logtheta3_loc,
        "logtheta4_loc": logtheta4_loc,
    }, os.path.join(params_dir, "logtheta_loc.pt"))

    torch.save({
        "organ_weight_1.weight": params_all["organ_weight_1.weight"],
        "organ_weight_1.bias": params_all["organ_weight_1.bias"],
        "organ_weight_2.weight": params_all["organ_weight_2.weight"],
        "organ_weight_2.bias": params_all["organ_weight_2.bias"],
        "organ_weight_3.weight": params_all["organ_weight_3.weight"],
        "organ_weight_3.bias": params_all["organ_weight_3.bias"],
    }, os.path.join(params_dir, "organ_weights.pt"))

    # Save train set morphologies and organs per task
    organ_logits_reshaped = organ_logits.reshape(-1, 25, args.organ_num)
    organs_type = torch.stack([
        torch.argmax(organ_logits_reshaped[i], dim=-1).reshape(5, 5)
        for i in range(organ_logits_reshaped.shape[0])
    ])
    operator_2 = operator(5)
    morphs = vec_to_morph(robos.float(), operator_2, 5)

    cumulative = [0] + list(accumulate(train_counts))

    train_dir = os.path.join(args.save_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    for t_idx in range(task_num):
        task_path = os.path.join(train_dir, task_dirs[t_idx])
        os.makedirs(task_path, exist_ok=True)
        start, end = cumulative[t_idx], cumulative[t_idx + 1]
        torch.save(morphs[start:end], os.path.join(task_path, "morph.pt"))
        torch.save(organs_type[start:end], os.path.join(task_path, "organ.pt"))

    # Inference on test set
    test_robos_list = []
    test_tasks_list = []
    test_morphs_list = []
    test_counts = []
    for t_idx, task_dir in enumerate(task_dirs):
        path = os.path.join(args.survivors_root, task_dir)
        task_oh = [0] * task_num
        task_oh[t_idx] = 1
        count = 0
        for j, fn in enumerate(per_task_files[t_idx]):
            if j not in train_indices[t_idx]:
                data = np.load(os.path.join(path, fn))["arr_0"]
                test_robos_list.append(morph_to_vec(data).reshape(1, 25 * 5))
                test_tasks_list.append(torch.tensor(task_oh).reshape(1, -1))
                test_morphs_list.append(torch.tensor(data).reshape(1, 5, 5))
                count += 1
        test_counts.append(count)

    if test_robos_list:
        test_robos = torch.cat(test_robos_list, dim=0)
        test_tasks = torch.cat(test_tasks_list, dim=0)
        test_morphs = torch.cat(test_morphs_list, dim=0)

        # Pad test set to match train set size for the guide input
        n_test = test_robos.shape[0]
        n_train = robos.shape[0]
        if n_test < n_train:
            padded_robos = torch.cat([test_robos, robos[:n_train - n_test]], dim=0)
            padded_tasks = torch.cat([test_tasks, tasks[:n_train - n_test]], dim=0)
        else:
            padded_robos = test_robos[:n_train]
            padded_tasks = test_tasks[:n_train]

        _, _, _, _, _, _, _, _, test_organ_logits = model.guide(
            padded_robos.float(), padded_tasks.float())
        test_organ_logits = test_organ_logits.reshape(-1, 25, args.organ_num)
        test_organs_type = torch.stack([
            torch.argmax(test_organ_logits[i], dim=-1).reshape(5, 5)
            for i in range(n_test)
        ])

        test_dir = os.path.join(args.save_dir, "test")
        os.makedirs(test_dir, exist_ok=True)
        cumulative_test = [0] + list(accumulate(test_counts))
        for t_idx in range(task_num):
            task_path = os.path.join(test_dir, task_dirs[t_idx])
            os.makedirs(task_path, exist_ok=True)
            start, end = cumulative_test[t_idx], cumulative_test[t_idx + 1]
            torch.save(test_morphs[start:end], os.path.join(task_path, "morph.pt"))
            torch.save(test_organs_type[start:end], os.path.join(task_path, "organ.pt"))

    print(f"Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
