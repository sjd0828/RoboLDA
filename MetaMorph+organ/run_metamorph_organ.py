import os, time, shutil, random, math, torch, sys, numpy as np

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))

from ppo.run_ppo_universal import run_ppo_att

from ppo.arguments import get_args

from evogym import sample_robot
from evogym.utils import is_connected, has_actuator, get_full_connectivity

import utils.mp_group as mp
from utils.algo_utils import TerminationCondition, Structure

from a2c_ppo_acktr.metamorph_organ import ATTBase


def run_universal(structures, organs, args, tc):
    
    save_path_controller = os.path.join(root_dir, "saved_data", name, "controller")
    
    transformer_mu = ATTBase(h_dim = args.h_dim, obs_dim = args.obs_dim, action_dim = args.action_dim, hidden_dim = args.hidden_dim,
                             body_size = 5, n_head = args.n_head, dropout = args.dropout)

    transformer_v = ATTBase(h_dim = args.h_dim, obs_dim = args.obs_dim, action_dim = 1, hidden_dim = args.hidden_dim,
                            body_size = 5, n_head = args.n_head, dropout = args.dropout)
                                
    group = mp.Group()   
    for structure, organ in zip(structures, organs):
        
        ppo_args = (2, args, "Carrier-v0", structure, organ, tc, (save_path_controller, structure.label), transformer_mu, transformer_v, output_dir) 
        
        group.add_job(run_ppo_att, ppo_args, callback = structure.set_reward)
    
    group.run_jobs(args.num_cores)
    for structure in structures:
        structure.compute_fitness()
    
    # record fitness
    f = open(temp_path, "a")
    out = ""
    for structure in structures:
        out += str(structure.label) + "\t\t" + str(structure.fitness) + "\n"
    f.write(out)
    f.close()
    
    
if __name__ == "__main__":
    
    # 备注是否为random_organ
    random_flag = False
    train_flag = False
    task = "carrier"
    if not random_flag:
        if train_flag:
            name = task+"-metamorph-lda-last-organ-train-" + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        else:
            name = task+"-metamorph-lda-last-organ-test-" + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    else:
        if train_flag:
            name = task+"-metamorph-rand-first-organ-train-" + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        else:
            name = task+"-metamorph-rand-first-organ-test-" + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    
    torch.multiprocessing.set_start_method("spawn")
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    
    args = get_args()
    args.train_iters = 2000
    # args.num_epochs = 200
    
    
    tc = TerminationCondition(args.train_iters)
    
    os.makedirs(os.path.join(root_dir, "saved_data", name))
    os.makedirs(os.path.join(root_dir, "saved_data", name, "controller"))
    temp_path = os.path.join(root_dir, "saved_data", name, "output.txt")
    output_dir = os.path.join(root_dir, "saved_data", name, "outputs")
    os.makedirs(output_dir)
    
    args.lr = 1e-4
    # args.clip_param = 0.1
    # args.num_mini_batch = 4
    # args.ppo_epoch = 4
    # args.entropy_coef = 0
    
    args.num_processes = 1
    args.num_steps = 128
    args.num_cores = 10
    args.width = 5
    args.eval_interval = 25
    args.log_interval = 5
    
    args.cuda = False
    args.no_cuda = True
    
    # h_dim: walker 2 pusher 6 carrier 6 catcher 7 climber 2 upstepper 14
    
    args.action_dim = 1
    args.obs_dim = 8 # 4*2
    args.h_dim = 6
    args.hidden_dim = 128
    args.n_head = 1
    args.dropout = 0.1
    
    if train_flag:
        robos = torch.load("train/0412morph&organ/"+task+"/morph.pt")
    else:
        robos = torch.load("test/0412morph&organ/"+task+"/morph.pt")
    
    indices = np.random.choice(robos.shape[0], size=10, replace=False) 
    
    structures = []
    for label in indices:
        temp_robo = np.array(robos[label])
        temp_structure = (temp_robo, get_full_connectivity(temp_robo))
        structures.append(Structure(*temp_structure, label, task_id=0))
    #print(structures[0])
    print("Structures loaded.")
    
    if random_flag == False:
        #organs = torch.load("robots/walker/organ_train.pt")[indices,:,:]
        if train_flag:
            origin_organs = torch.load("train/0412morph&organ/"+task+"/organ.pt")[indices,:,:]
        else:
            origin_organs = torch.load("test/0412morph&organ/"+task+"/organ.pt")[indices,:,:]
        print(origin_organs.shape)
        organs = torch.zeros(origin_organs.shape[0], 10, 25)
        for r in range(origin_organs.shape[0]):
            for i in range(5):
                for j in range(5):
                    organs[r,origin_organs[r,i,j].long(),i*5+j] = 1
        torch.save(origin_organs,root_dir+"/saved_data/"+name+'/lda_organ.pt')
    #print(organs.shape)

    else:
        organ_rand = torch.zeros(indices.shape[0], 10, 25)
        organs_true = torch.zeros(indices.shape[0],5,5)
        for r in range(indices.shape[0]):
            for i in range(5):
                for j in range(5):
                    t = random.randint(0,3)
                    organ_rand[r,t,i*5+j] = 1
                    organs_true[r,i,j] = t
        organs = organ_rand
        torch.save(organs_true,root_dir+"/saved_data/"+name+'/random_organ.pt')
    
    run_universal(structures, organs, args, tc)
    
