import os, sys
sys.path.insert(1, os.path.join(sys.path[0], 'externals', 'pytorch_a2c_ppo_acktr_gail'))

import numpy as np
import time
from collections import deque
import torch

from ppo import utils
from ppo.arguments import get_args
from ppo.evaluate import evaluate
from ppo.envs import make_vec_envs

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.algo.ppo import PPO
from a2c_ppo_acktr.algo import gail
#from a2c_ppo_acktr.model import Policy
#from a2c_ppo_acktr.transformer_hypernet_monolithic import Policy
from a2c_ppo_acktr.metamorph import Policy
from a2c_ppo_acktr.storage import RolloutStorage

import evogym.envs

import time

# Derived from
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

def run_ppo_att(
    idd,
    args, # ppo arguments
    env_name, # task name
    structure, # robot structure
    termination_condition, 
    saving_convention, 
    transformer_mu, # shared actor
    transformer_v,  # shared critic
    output_dir, # save evaluated fitness per iter 
    override_env_name = None,
    verbose = True):

    assert (structure == None) == (termination_condition == None) and (structure == None) == (saving_convention == None)

    print(f'Starting training on \n{structure}\nat {saving_convention}...\n')

    #args = get_args()
    #if override_env_name:
    #    args.env_name = override_env_name
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = args.log_dir
    if saving_convention != None:
        log_dir = os.path.join(saving_convention[0], log_dir, "task-{}_robot-{}".format(structure.task_id, str(saving_convention[1])))
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    #device = torch.device("cuda:0" if args.cuda else "cpu")
    device = torch.device("cuda:{}".format(idd%4) if args.cuda else "cpu")

    #envs = make_vec_envs(env_name, structure, args.seed, args.num_processes,
    #                     args.gamma, args.log_dir, device, False)
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
                  "DownStepper-v0":14,
                  "BridgeWalker-v0":3}
    head_size = head_sizes[env_name]

    actor_critic = Policy(body_size = args.width, structure = structure,
                         transformer_mu = transformer_mu, transformer_v = transformer_v, global_size = head_size)
    actor_critic.to(device)

    if args.algo == 'a2c':
        print('Warning: this code has only been tested with ppo.')
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
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
    elif args.algo == 'acktr':
        print('Warning: this code has only been tested with ppo.')
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                env_name.split('-')[0].lower()))

        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              100) # 100: recurrent_state_hidden_size is not used

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
    
    output_path = os.path.join(output_dir, "output"+str(saving_convention[1])+".txt")
    
    
    # zero-shot performance
    obs_rms = utils.get_vec_normalize(envs).obs_rms
    determ_avg_reward = evaluate(args.num_evals, actor_critic, obs_rms, env_name, (structure.body, structure.connections), 
                                  args.seed, args.num_processes, eval_log_dir, device)
    out = str(saving_convention[1]) + "\t\t" + "-1" + "\t\t" + str(determ_avg_reward) + "\n"
        
    f = open(output_path, "a")
    f.write(out)
    f.close()
    
    
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)
        
        start = time.time()
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                

                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            # track rewards
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    rewards_tracker.append(info['episode']['r'])
                    if len(rewards_tracker) < 10:
                        avg_rewards_tracker.append(np.average(np.array(rewards_tracker)))
                    else:
                        avg_rewards_tracker.append(np.average(np.array(rewards_tracker[-10:])))

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
        end = time.time()
        # print("sampling:", end-start)
        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        print("updating modular policy for robot ", str(structure.label))
        #print("=============", agent.actor_critic.transformer.encoder.weight)
        start = time.time()
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        end = time.time()
        # print("update:",end-start)
        rollouts.after_update()
     
        # print status
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
        
        # evaluate the controller and save it if it does the best so far
        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            determ_avg_reward = evaluate(args.num_evals, actor_critic, obs_rms, env_name, (structure.body, structure.connections), 
                                         args.seed, args.num_processes, eval_log_dir, device)

            if verbose:
                if saving_convention != None:
                    print(f'In task {structure.task_id}, evaluated {saving_convention[1]} using {args.num_evals} episodes. Mean reward: {np.mean(determ_avg_reward)}\n')
                else:
                    print(f'Evaluated using {args.num_evals} episodes. Mean reward: {np.mean(determ_avg_reward)}\n')

            if determ_avg_reward > max_determ_avg_reward:
                max_determ_avg_reward = determ_avg_reward

                temp_path = os.path.join(args.save_dir, args.algo, env_name + ".pt")
                if saving_convention != None:
                    temp_path = os.path.join(saving_convention[0], "task-{}_robot-{}".format(structure.task_id, str(saving_convention[1])) + "_controller" + ".pt")
                
                if verbose:
                    print(f'Saving {temp_path} with avg reward {max_determ_avg_reward}\n')
                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
                ], temp_path)
        
            # save fitness 
            out = str(saving_convention[1]) + "\t\t" + str(j) + "\t\t" + str(determ_avg_reward) + "\n"
        
        '''
        else:
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            determ_avg_reward = evaluate(args.num_evals, actor_critic, obs_rms, env_name, (structure.body, structure.connections), 
                                         args.seed, args.num_processes, eval_log_dir, device)
            out = str(saving_convention[1]) + "\t\t" + str(j) + "\t\t" + str(determ_avg_reward) + "\n"
        '''
        f = open(output_path, "a")
        f.write(out)
        f.close()
    
        # return upon reaching the termination condition
        if not termination_condition == None:
            if termination_condition(j):
                if verbose:
                    print(f'{saving_convention} has met termination condition ({j})...terminating...\n')
                return max_determ_avg_reward

#python ppo_main_test.py --env-name "roboticgamedesign-v0" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01
#python ppo.py --env-name "roboticgamedesign-v0" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --log-dir "logs/"
