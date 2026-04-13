"""Zero-shot morphology generation for an unseen task using trained RoboLDA."""

import argparse
import os

import numpy as np
import pyro
import torch
import torch.nn.functional as F
from pyro.infer import config_enumerate

from VAE_RoboLDA import RoboLDA
from vec2morph import operator, vec_to_morph


def parse_args():
    parser = argparse.ArgumentParser(description="Generate morphologies for a target task")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to params/ directory from training")
    parser.add_argument("--num-generate", type=int, default=25)
    parser.add_argument("--save-dir", type=str, default="generated")
    parser.add_argument("--individual-num", type=int, default=6)
    parser.add_argument("--organ-num", type=int, default=6)
    parser.add_argument("--comp-num", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def is_connected(body):
    """Check if all non-empty voxels form a single connected component."""
    visited = set()
    start = None
    for i in range(body.shape[0]):
        for j in range(body.shape[1]):
            if body[i, j] != 0:
                start = (i, j)
                break
        if start:
            break
    if start is None:
        return False
    stack = [start]
    while stack:
        x, y = stack.pop()
        if (x, y) in visited:
            continue
        if x < 0 or x >= body.shape[0] or y < 0 or y >= body.shape[1]:
            continue
        if body[x, y] == 0:
            continue
        visited.add((x, y))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            stack.append((x + dx, y + dy))
    for i in range(body.shape[0]):
        for j in range(body.shape[1]):
            if body[i, j] != 0 and (i, j) not in visited:
                return False
    return True


def has_actuator(body):
    return np.any((body == 3) | (body == 4))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    params = torch.load(os.path.join(args.model_dir, "logtheta_loc.pt"),
                        map_location="cpu")

    logtheta3 = params["logtheta3_loc"]
    logtheta4 = params["logtheta4_loc"]

    individual_num = args.individual_num
    organ_num = args.organ_num
    comp_num = args.comp_num

    # Only transfer organ-voxel layers (θ₃, θ₄); θ₁/θ₂ are sampled from prior
    theta3 = F.softmax(logtheta3.reshape(comp_num, individual_num, organ_num), -1)
    theta4 = F.softmax(logtheta4.reshape(25, organ_num, 5), -1)

    organ_weights_path = os.path.join(args.model_dir, "organ_weights.pt")
    if os.path.exists(organ_weights_path):
        organ_weights = torch.load(organ_weights_path, map_location="cpu")
    else:
        organ_weights = None

    operator_2 = operator(5)
    os.makedirs(args.save_dir, exist_ok=True)

    voxel_location = torch.zeros(25, 10)
    for r in range(5):
        for c in range(5):
            idx = r * 5 + c
            voxel_location[idx, r] = 1
            voxel_location[idx, 5 + c] = 1

    generated = []
    seen_hashes = set()
    attempts = 0
    max_attempts = args.num_generate * 100

    while len(generated) < args.num_generate and attempts < max_attempts:
        attempts += 1

        # θ₂ sampled from prior N(0,I), then softmax -> random individual distribution
        raw_theta2 = torch.randn(individual_num)
        ind_probs = F.softmax(raw_theta2, dim=-1).unsqueeze(0)
        ind = torch.distributions.OneHotCategorical(probs=ind_probs).sample().squeeze(0)

        # Compute organ assignment probabilities
        if organ_weights is not None:
            temp = torch.cat([
                ind.unsqueeze(0).repeat(25, 1),
                voxel_location
            ], dim=1)
            ow1_w = organ_weights["organ_weight_1.weight"]
            ow1_b = organ_weights["organ_weight_1.bias"]
            ow2_w = organ_weights["organ_weight_2.weight"]
            ow2_b = organ_weights["organ_weight_2.bias"]
            ow3_w = organ_weights["organ_weight_3.weight"]
            ow3_b = organ_weights["organ_weight_3.bias"]
            h = torch.tanh(F.linear(temp, ow1_w, ow1_b))
            h = torch.tanh(F.linear(h, ow2_w, ow2_b))
            ow = F.linear(h, ow3_w, ow3_b)  # (25, comp_num)

            t3 = theta3.reshape(comp_num, -1)  # (comp_num, ind*organ)
            organ_logits = torch.mm(ow, t3).reshape(25, individual_num, organ_num)
            organ_logits = torch.bmm(
                ind.unsqueeze(0).repeat(25, 1).unsqueeze(1),
                organ_logits
            ).squeeze(1)  # (25, organ_num)
        else:
            organ_logits = torch.ones(25, organ_num)

        organs = torch.distributions.OneHotCategorical(logits=organ_logits).sample()

        # Sample voxels
        voxel_params_list = []
        for pos in range(25):
            organ_vec = organs[pos]  # (organ_num,)
            vp = torch.mm(organ_vec.unsqueeze(0), theta4[pos])  # (1, 5)
            voxel_params_list.append(vp)
        voxel_params = torch.cat(voxel_params_list, dim=0)  # (25, 5)

        voxels = torch.distributions.OneHotCategorical(probs=voxel_params).sample()
        body_vec = voxels.reshape(1, 125)
        body = vec_to_morph(body_vec.float(), operator_2, 5).numpy().astype(int)[0]

        if not is_connected(body) or not has_actuator(body):
            continue

        body_hash = body.tobytes()
        if body_hash in seen_hashes:
            continue
        seen_hashes.add(body_hash)

        save_path = os.path.join(args.save_dir, f"gen_{len(generated):04d}.npz")
        np.savez(save_path, body)
        generated.append(body)

    print(f"Generated {len(generated)} valid morphologies ({attempts} attempts)")


if __name__ == "__main__":
    main()
