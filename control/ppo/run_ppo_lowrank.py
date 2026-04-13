import os, sys
sys.path.insert(1, os.path.join(sys.path[0], 'externals', 'pytorch_a2c_ppo_acktr_gail'))

import numpy as np
import time
from collections import deque
import torch

from ppo import utils
from ppo.arguments import get_args
from ppo.evaluate_universal import evaluate
from ppo.evaluate_universal_organ import evaluate as evaluate_organ
from ppo.envs import make_vec_envs

from a2c_ppo_acktr.algo.ppo_universal import PPO
from a2c_ppo_acktr.metamorph_organ import Policy
from a2c_ppo_acktr.storage import RolloutStorage

import evogym.envs

from sklearn.cluster import AffinityPropagation


def run_ppo_att(
    idd,
    args,
    env_name,
    structure,
    organs,
    termination_condition,
    saving_convention,
    transformer_mu,
    transformer_v,
    output_dir,
    override_env_name=None,
    verbose=True):

    assert (structure is None) == (termination_condition is None) and (structure is None) == (saving_convention is None)

    print(f'Starting training on \n{structure}\nat {saving_convention}...\n')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = args.log_dir
    if saving_convention is not None:
        log_dir = os.path.join(saving_convention[0], log_dir, "task-{}_robot-{}".format(structure.task_id, str(saving_convention[1])))
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:{}".format(idd % 4) if args.cuda else "cpu")

    envs = make_vec_envs(env_name, (structure.body, structure.connections), args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    head_sizes = {"Walker-v0": 2,
                  "Carrier-v0": 6,
                  "Pusher-v0": 6,
                  "Balancer-v0": 1,
                  "Thrower-v0": 6,
                  "Catcher-v0": 7,
                  "Climber-v0": 2,
                  "UpStepper-v0": 14,
                  "DownStepper-v0": 14,
                  "BridgeWalker-v0": 3}
    head_size = head_sizes[env_name]

    actor_critic = Policy(global_size=head_size, body_size=args.width, organs=organs, structure=structure,
                          transformer_mu=transformer_mu, transformer_v=transformer_v)

    # initialize organs as full-connection
    actor_critic.organs = torch.zeros(5, 25)
    actor_critic.organs[0, :] = 1

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
        max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              100)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    rewards_tracker = []
    avg_rewards_tracker = []
    sliding_window_size = 10
    max_determ_avg_reward = float('-inf')

    output_path = os.path.join(output_dir, "output" + str(saving_convention[1]) + ".txt")

    try:
        os.mkdir(os.path.join(output_dir, "organs"))
    except OSError:
        pass
    organ_path = os.path.join(output_dir, "organs", str(structure.label))
    os.mkdir(organ_path)

    # zero-shot performance
    obs_rms = utils.get_vec_normalize(envs).obs_rms
    actor_critic.eval()
    determ_avg_reward = evaluate(actor_critic.organs, 1, actor_critic, obs_rms, env_name,
                                 (structure.body, structure.connections),
                                 args.seed, args.num_processes, eval_log_dir, device)
    actor_critic.train()
    out = str(saving_convention[1]) + "\t\t" + "-1" + "\t\t" + str(determ_avg_reward) + "\n"

    f = open(output_path, "a")
    f.write(out)
    f.close()

    is_actuator = structure.body.flatten() >= 3

    # compute initial distance-based affinity matrix
    if len(os.listdir(organ_path)) == 0:
        affinity = np.zeros((25, 25))
        for x in range(25):
            ix = x // 5
            jx = x % 5
            for y in range(x + 1, 25):
                iy = y // 5
                jy = y % 5
                affinity[x, y] = np.exp(-1 * (abs(ix - iy) + abs(jx - jy)))
                affinity[y, x] = np.exp(-1 * (abs(ix - iy) + abs(jx - jy)))
        np.save(os.path.join(organ_path, "affinity.npy"), affinity)
    else:
        affinity = np.load(os.path.join(organ_path, "affinity.npy"))

    for j in range(num_updates):

        if args.use_linear_lr_decay:
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], actor_critic.organs)

            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    rewards_tracker.append(info['episode']['r'])
                    if len(rewards_tracker) < 10:
                        avg_rewards_tracker.append(np.average(np.array(rewards_tracker)))
                    else:
                        avg_rewards_tracker.append(np.average(np.array(rewards_tracker[-10:])))

            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1], actor_critic.organs).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts, actor_critic.organs)
        rollouts.after_update()

        if j % args.log_interval == 0 and len(episode_rewards) > 1 and verbose:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):

            actor_critic.eval()

            obs_rms = utils.get_vec_normalize(envs).obs_rms
            determ_avg_reward = evaluate(actor_critic.organs, 1, actor_critic, obs_rms, env_name,
                                         (structure.body, structure.connections),
                                         args.seed, args.num_processes, eval_log_dir, device)

            # update organs via AP clustering every 100 iterations
            if j % 100 == 0 and j > 0:
                a = 0
                gaps = []
                bs = []
                for b, bb in enumerate(is_actuator):
                    if bb:
                        determ_avg_reward_ = evaluate_organ(a, actor_critic.organs, 1, actor_critic, obs_rms, env_name,
                                                            (structure.body, structure.connections),
                                                            args.seed, 1, eval_log_dir, device)
                        gaps.append(determ_avg_reward - determ_avg_reward_)
                        a += 1
                        bs.append(b)
                preference = [0] * 25
                for a, b in enumerate(bs):
                    preference[b] = gaps[a]
                preference = np.array(preference)
                preference = np.exp(preference) / np.sum(np.exp(preference))

                AP = AffinityPropagation(preference=preference, affinity="precomputed",
                                         random_state=928, verbose=False, damping=0.9, max_iter=1000).fit(affinity)
                labels = AP.labels_
                num_clusters = np.max(labels) + 1
                organs_ = np.zeros((num_clusters, 25))
                for c in range(num_clusters):
                    organs_[c, labels == c] = 1
                actor_critic.organs = torch.tensor(organs_).to(device)
                np.save(os.path.join(organ_path, "organ_" + str(j) + ".npy"), organs_)
                print("organ updated!")

            if verbose:
                if saving_convention is not None:
                    print(f'In task {structure.task_id}, evaluated {saving_convention[1]} using {args.num_evals} episodes. Mean reward: {np.mean(determ_avg_reward)}\n')
                else:
                    print(f'Evaluated using {args.num_evals} episodes. Mean reward: {np.mean(determ_avg_reward)}\n')

            if determ_avg_reward > max_determ_avg_reward:
                max_determ_avg_reward = determ_avg_reward

                temp_path = os.path.join(args.save_dir, args.algo, env_name + ".pt")
                if saving_convention is not None:
                    temp_path = os.path.join(saving_convention[0], "task-{}_robot-{}".format(structure.task_id, str(saving_convention[1])) + "_controller" + ".pt")

                if verbose:
                    print(f'Saving {temp_path} with avg reward {max_determ_avg_reward}\n')
                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
                ], temp_path)

            out = str(saving_convention[1]) + "\t\t" + str(j) + "\t\t" + str(determ_avg_reward) + "\n"
            actor_critic.train()

        f = open(output_path, "a")
        f.write(out)
        f.close()

        if termination_condition is not None:
            if termination_condition(j):
                if verbose:
                    print(f'{saving_convention} has met termination condition ({j})...terminating...\n')
                return max_determ_avg_reward
