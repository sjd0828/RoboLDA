"""NEAT/CPPN generation-0 population (only requires neat-python when imported by eval_ea_gen0_baselines)."""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import neat
import numpy as np
import torch

_EXAMPLES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXTERNAL_PPO = os.path.join(_EXAMPLES, "externals", "pytorch_a2c_ppo_acktr_gail")
_EXTERNAL_NEAT = os.path.join(_EXAMPLES, "externals", "PyTorch-NEAT")
for _p in (_EXAMPLES, _EXTERNAL_PPO, _EXTERNAL_NEAT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evogym import get_full_connectivity, hashable, is_connected, has_actuator
from pytorch_neat.cppn import create_cppn
from utils.algo_utils import Structure

from cppn_neat.population import Population as CppnPopulation

MAX_NEAT_DRAWS = 20000


def _meshgrid_2d(structure_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        x, y = torch.meshgrid(
            torch.arange(structure_shape[0]),
            torch.arange(structure_shape[1]),
            indexing="ij",
        )
    except TypeError:
        x, y = torch.meshgrid(torch.arange(structure_shape[0]), torch.arange(structure_shape[1]))
    x, y = x.flatten(), y.flatten()
    center = (np.array(structure_shape) - 1) / 2
    d = ((x - center[0]) ** 2 + (y - center[1]) ** 2).sqrt()
    return x, y, d


def cppn_robot_from_genome(genome: neat.DefaultGenome, config: neat.Config) -> np.ndarray:
    nodes = create_cppn(
        genome,
        config,
        leaf_names=["x", "y", "d"],
        node_names=["empty", "rigid", "soft", "hori", "vert"],
    )
    structure_shape = config.extra_info["structure_shape"]
    x, y, d = _meshgrid_2d(structure_shape)
    material = []
    for node in nodes:
        material.append(node(x=x, y=y, d=d).numpy())
    material = np.vstack(material).argmax(axis=0)
    return material.reshape(structure_shape)


def _cppn_constraint_single(
    genome: neat.DefaultGenome,
    config: neat.Config,
    structure_hashes: Dict,
) -> Tuple[bool, bool]:
    robot = cppn_robot_from_genome(genome, config)
    validity = is_connected(robot) and has_actuator(robot)
    valid = validity
    if validity:
        rh = hashable(robot)
        if rh in structure_hashes:
            validity = False
        else:
            structure_hashes[rh] = True
    return validity, valid


def cppn_constraint_batch(
    genomes: List[Tuple[int, neat.DefaultGenome]],
    config: neat.Config,
    _generation: int,
    structure_hashes: Dict,
) -> List[Tuple[bool, bool]]:
    return [_cppn_constraint_single(g, config, structure_hashes) for _, g in genomes]


def fill_cppn_valid_population(
    pop: CppnPopulation,
    config: neat.Config,
    structure_hashes: Dict,
) -> None:
    generation = pop.generation
    genomes = list(pop.population.items())

    def batch_fn(g_list, cfg, gen):
        return cppn_constraint_batch(g_list, cfg, gen, structure_hashes)

    validity_2 = batch_fn(genomes, config, generation)
    validity = [validity_2[kk][0] for kk in range(len(validity_2))]
    valid_list = [genomes[i] for i in range(len(genomes)) if validity[i]]

    n_draws = 0
    while len(valid_list) < config.pop_size and n_draws < MAX_NEAT_DRAWS:
        need = config.pop_size - len(valid_list)
        new_population = pop.reproduction.create_new(
            config.genome_type,
            config.genome_config,
            need,
        )
        new_genomes = list(new_population.items())
        n_draws += len(new_genomes)

        validity_2 = batch_fn(new_genomes, config, generation)
        validity = [validity_2[kk][0] for kk in range(len(validity_2))]
        for i in range(len(new_genomes)):
            if validity[i]:
                valid_list.append(new_genomes[i])

    if len(valid_list) < config.pop_size:
        raise RuntimeError(
            f"NEAT: could not fill pop_size={config.pop_size} after {n_draws} draft genomes"
        )

    pop.population = dict(valid_list[: config.pop_size])
    pop.species.speciate(config, pop.population, generation)


def build_cppn_gen0_structures(
    neat_cfg_path: str,
    pop_size: int,
    shape: Tuple[int, int],
    save_dir: str,
    target_task: str,
    train_iters: int,
) -> Tuple[List[Structure], List[str]]:
    """Build NEAT generation-0 (random CPPN) Structure list and logging labels."""
    structure_hashes: Dict = {}
    n_cfg = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        neat_cfg_path,
        extra_info={
            "structure_shape": shape,
            "train_iters": train_iters,
            "save_path": save_dir,
            "structure_hashes": structure_hashes,
            "env_name": target_task,
        },
        custom_config=[
            ("NEAT", "pop_size", pop_size),
        ],
    )
    pop = CppnPopulation(n_cfg)
    fill_cppn_valid_population(pop, n_cfg, structure_hashes)

    struct_dir = os.path.join(save_dir, "structures")
    os.makedirs(struct_dir, exist_ok=True)
    ordered = sorted(pop.population.items(), key=lambda kv: kv[0])
    structures: List[Structure] = []
    labels: List[str] = []
    for i, (gid, genome) in enumerate(ordered):
        body = cppn_robot_from_genome(genome, n_cfg)
        conn = get_full_connectivity(body)
        pth = os.path.join(struct_dir, f"{i}_gid{gid}.npz")
        np.savez(pth, body, conn)
        structures.append(Structure(body, conn, i, 0))
        labels.append(f"cppn_gen0/gid_{gid}")
    return structures, labels
