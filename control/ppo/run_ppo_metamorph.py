"""PPO training loop for vanilla MetaMorph (no organ masks).

Policy.act(obs) and Policy.evaluate_actions(obs, actions) do NOT take organs.
Uses the base PPO from a2c_ppo_acktr.algo.ppo (update without organs).
"""

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], 'externals', 'pytorch_a2c_ppo_acktr_gail'))

import numpy as np
import time
from collections import deque
import torch

from ppo import utils
from ppo.arguments import get_args
from ppo.evaluate_metamorph import evaluate
from ppo.envs import make_vec_envs

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.algo.ppo import PPO
from a2c_ppo_acktr.metamorph import Policy
from a2c_ppo_acktr.storage import RolloutStorage

from evogym import get_full_connectivity
import evogym.envs

HEAD_SIZES = {
    "Walker-v0": 2, "Carrier-v0": 6, "Pusher-v0": 6,
    "Climber-v0": 2, "UpStepper-v0": 14, "DownStepper-v0": 14,
    "BridgeWalker-v0": 3, "Catcher-v0": 7, "Balancer-v0": 1, "Thrower-v0": 6,
}


def run_ppo_att(
    idd,
    args,
    env_name,
    structure,
    termination_condition,
    saving_convention,
    transformer_mu,
    transformer_v,
    output_dir,
    verbose=True,
):
    assert (structure is None) == (termination_condition is None)
    assert (structure is None) == (saving_convention is None)

    print(f'Starting training on \n{structure}\nat {saving_convention}...\n')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = args.log_dir
    if saving_convention is not None:
        log_dir = os.path.join(
            saving_convention[0], log_dir,
            "task-{}_robot-{}".format(structure.task_id, str(saving_convention[1])),
        )
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    num_gpus = max(1, torch.cuda.device_count()) if args.cuda else 1
    device = torch.device("cuda:{}".format(idd % num_gpus) if args.cuda else "cpu")

    envs = make_vec_envs(
        env_name,
        (structure.body, structure.connections),
        args.seed, args.num_processes,
        args.gamma, args.log_dir, device, False,
    )

    head_size = HEAD_SIZES[env_name]

    actor_critic = Policy(
        body_size=args.width, structure=structure,
        transformer_mu=transformer_mu, transformer_v=transformer_v,
        global_size=head_size,
    )
    actor_critic.to(device)

    agent = PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
    )

    rollouts = RolloutStorage(
        args.num_steps, args.num_processes,
        envs.observation_space.shape, envs.action_space, 100,
    )

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    max_determ_avg_reward = float('-inf')

    output_path = os.path.join(output_dir, "output{}.txt".format(saving_convention[1]))

    obs_rms = utils.get_vec_normalize(envs).obs_rms
    determ_avg_reward = evaluate(
        args.num_evals, actor_critic, obs_rms, env_name,
        (structure.body, structure.connections),
        args.seed, args.num_processes, eval_log_dir, device,
    )
    with open(output_path, "a") as f:
        f.write("{}\t\t{}\t\t{}\n".format(saving_convention[1], -1, determ_avg_reward))

    for j in range(num_updates):
        if args.use_linear_lr_decay:
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr,
            )

        for step in range(args.num_steps):
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step])

            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1]).detach()

        rollouts.compute_returns(
            next_value, args.use_gae, args.gamma,
            args.gae_lambda, args.use_proper_time_limits,
        )

        if verbose:
            print("updating modular policy for robot ", str(structure.label))
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        if j % args.log_interval == 0 and len(episode_rewards) > 1 and verbose:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print(
                "Updates {}, num timesteps {} \n "
                "Last {} training episodes: mean/median reward {:.1f}/{:.1f}, "
                "min/max reward {:.1f}/{:.1f}\n".format(
                    j, total_num_steps,
                    len(episode_rewards), np.mean(episode_rewards),
                    np.median(episode_rewards), np.min(episode_rewards),
                    np.max(episode_rewards)))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            determ_avg_reward = evaluate(
                args.num_evals, actor_critic, obs_rms, env_name,
                (structure.body, structure.connections),
                args.seed, args.num_processes, eval_log_dir, device,
            )

            if verbose:
                print(f'Evaluated {saving_convention[1]} using {args.num_evals} episodes. '
                      f'Mean reward: {np.mean(determ_avg_reward)}\n')

            if determ_avg_reward > max_determ_avg_reward:
                max_determ_avg_reward = determ_avg_reward
                temp_path = os.path.join(
                    saving_convention[0],
                    "task-{}_robot-{}_controller.pt".format(
                        structure.task_id, str(saving_convention[1])),
                )
                if verbose:
                    print(f'Saving {temp_path} with avg reward {max_determ_avg_reward}\n')
                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'obs_rms', None),
                ], temp_path)

            with open(output_path, "a") as f:
                f.write("{}\t\t{}\t\t{}\n".format(
                    saving_convention[1], j, determ_avg_reward))

        if termination_condition is not None:
            if termination_condition(j):
                if verbose:
                    print(f'{saving_convention} has met termination condition ({j})...terminating...\n')
                return max_determ_avg_reward

    return max_determ_avg_reward
