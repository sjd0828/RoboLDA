import os
import sys
import numpy as np
import time
from collections import deque
import torch

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, "..")
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(root_dir, "externals", "pytorch_a2c_ppo_acktr_gail"))

from ppo import utils
from ppo.arguments import get_args
from ppo.evaluate import evaluate
from ppo.envs import make_vec_envs

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

import evogym.envs


def run_ppo(
    idd,
    args,
    env_name,
    structure,
    termination_condition,
    saving_convention,
    override_env_name=None,
    verbose=True,
):
    print(f"Starting training on \n{structure}\nat {saving_convention}...\n")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = args.log_dir
    if saving_convention is not None:
        log_dir = os.path.join(
            saving_convention[0], log_dir,
            "task-{}_robot-{}".format(structure.task_id, str(saving_convention[1])))
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:{}".format(idd % 4) if args.cuda else "cpu")

    envs = make_vec_envs(
        env_name, (structure.body, structure.connections),
        args.seed, args.num_processes, args.gamma, args.log_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={"recurrent": args.recurrent_policy})
    actor_critic.to(device)

    agent = algo.PPO(
        actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
        args.value_loss_coef, args.entropy_coef, lr=args.lr, eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(
        args.num_steps, args.num_processes,
        envs.observation_space.shape, envs.action_space,
        actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    max_determ_avg_reward = float("-inf")

    output_dir = os.path.join(saving_convention[0], "outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"output{saving_convention[1]}.txt")

    obs_rms = utils.get_vec_normalize(envs).obs_rms
    determ_avg_reward = evaluate(
        args.num_evals, actor_critic, obs_rms, env_name,
        (structure.body, structure.connections),
        args.seed, args.num_processes, eval_log_dir, device)
    out = f"{saving_convention[1]}\t\t-1\t\t{determ_avg_reward}\n"
    with open(output_path, "a") as f:
        f.write(out)

    for j in range(num_updates):
        if args.use_linear_lr_decay:
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if "episode" in info:
                    episode_rewards.append(info["episode"]["r"])

            masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if "bad_transition" in info else [1.0] for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        if j % args.log_interval == 0 and len(episode_rewards) > 1 and verbose:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print(
                f"Updates {j}, num timesteps {total_num_steps}, "
                f"mean/median reward {np.mean(episode_rewards):.1f}/{np.median(episode_rewards):.1f}")

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            determ_avg_reward = evaluate(
                args.num_evals, actor_critic, obs_rms, env_name,
                (structure.body, structure.connections),
                args.seed, args.num_processes, eval_log_dir, device)

            if determ_avg_reward > max_determ_avg_reward:
                max_determ_avg_reward = determ_avg_reward
                temp_path = os.path.join(
                    saving_convention[0],
                    f"task-{structure.task_id}_robot-{saving_convention[1]}_controller.pt")
                torch.save([actor_critic, getattr(utils.get_vec_normalize(envs), "obs_rms", None)],
                           temp_path)

            out = f"{saving_convention[1]}\t\t{j}\t\t{determ_avg_reward}\n"
            with open(output_path, "a") as f:
                f.write(out)

        if termination_condition is not None and termination_condition(j):
            return max_determ_avg_reward
