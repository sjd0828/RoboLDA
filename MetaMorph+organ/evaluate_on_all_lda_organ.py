import os, time, shutil, random, math, torch, sys, numpy as np
sys.path.insert(1, os.path.join("../", 'externals', 'pytorch_a2c_ppo_acktr_gail'))

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))

from ppo.run_hyper import run_ppo_att

from ppo.arguments import get_args

from evogym import sample_robot
from evogym.utils import is_connected, has_actuator, get_full_connectivity

import utils.mp_group as mp
from utils.algo_utils import TerminationCondition, Structure

from a2c_ppo_acktr.metamorph import ATTBase, Policy
import a2c_ppo_acktr
from evaluate import evaluate

#transformer_mu = torch.load("rand_mu.pt")
#transformer_v = torch.load("rand_v.pt")

task = "walker"
robos = torch.load("test/0412morph&organ/"+task+"/morph.pt")
#robos = torch.load("test/0702test/"+task+"/morph.pt")
#robos = torch.load("test/walker/morph.pt")

# [45,59,7,50,92,27,131,137,122,8]
#indices = [1,6,8,9,13,4,2,14,10,7]
#indices = [3,13,7,2,6,10,4,1,14,0]
#indices = [12,4,5,0,9,3,1,10,7,14]
#indices = [10,12,0,13,5,9,1,2,8,14]

#indices = [7,8,27,45,50,59,92,122,131,137]
#indices = [5,19,29,42,58,59,65,73,83,106]
#indices = [2,3,5,24,44,48,65,114,120,134]

#indices = [18,34,58,62,151,155,165,186,200,202]
#indices = [5,12,33,80,83,90,116,122,131,162]

#indices = [0,3,4,6,9,12,14,16,20,21]
#indices = [2,3,4,10,13,16,17,18,19,20]
#indices = [1,8,10,11,13,14,16,20,21,22]
indices = [14]
#indices = [19]

#indices = [0,1,2,3,4,5,6,7,8,9]
#indices = [9]

#indices = [1,6,8,10,11,13,14,18,19,20]

# enumerate([59,7,50,92,27,131,137,122,8])

stds = []
srs = []

for ii, label in enumerate(indices):
    robo = robos[label]
    temp_robo = np.array(robo)
    temp_structure = (temp_robo, get_full_connectivity(temp_robo))
    structure = Structure(*temp_structure, label, task_id=0)
    
    origin_organ = torch.load("test/0412morph&organ/"+task+"/organ.pt")[label,:,:]
    #origin_organ = torch.load("test/0702test/"+task+"/organ.pt")[label,:,:]
    #origin_organ = torch.load("test/walker/organ.pt")[label,:,:]
    #origin_organ = torch.load("controllers/carrier/test/rand-organ/4/random_organ.pt")[ii,:,:]
    
    organ = torch.zeros(6, 25)
    for i in range(5):
        for j in range(5):
            organ[origin_organ[i,j].long(),i*5+j] = 1 
    
    '''
    origin_organ = np.load("controllers/walker/test/lowrank2/1/organs/"+str(label)+"/organ_2000.npy")
    organ = torch.zeros(6, 25)
    for i in range(5):
        for j in range(5):
            organ[int(origin_organ[0,i*5+j]),i*5+j] = 1 
    '''
    #print(organ)
    actor_critic = torch.load("controllers/walker/test/no-organ/3/task-0_robot-"+str(label)+"_controller.pt")[0].to("cuda:0")
    obs_rms = torch.load("controllers/walker/test/no-organ/3/task-0_robot-"+str(label)+"_controller.pt")[1]
    actor_critic.eval()
    # actor_critic = Policy(body_size = 5, structure = structure,
    # transformer_mu = transformer_mu, transformer_v = transformer_v, global_size = 2)
    
                         
    action_std, sr = evaluate(organ, 1, actor_critic, obs_rms, "Walker-v0", (structure.body, structure.connections), 0, 1, "gym_eval", "cuda:0")
    #print(actions)
    #print(determ_avg_reward)
    print(label, action_std, sr)
    stds.append(action_std)
    srs.append(sr)
print("average: ", np.mean(np.array(stds)), np.mean(np.array(srs)))


