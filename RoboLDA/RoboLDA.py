import math, pandas as pd, numpy as np, pyro, torch
import random
import torch.nn as nn
import torch.nn.functional as F
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.optim import ClippedAdam, Adam
import pyro.distributions as dist
from tqdm import trange
import os
from vec2morph import vec_to_morph, morph_to_vec, operator
from VAE_RoboLDA import RoboLDA
from itertools import accumulate

random.seed(111)
root_path = 'survivors/'
sub_path = ['walker','pusher','carrier']
task_num = len(sub_path)
# inhomogeneous distribution parameters of task-ind, ind-organ and organ-voxel

from utils import is_connected, has_actuator, get_full_connectivity
#from algo_utils import Structure

flag = True
save_root_path = 'results/'

### Divide high-performance morphology into train set and test set
task_nums_all = []
for i, path_name in enumerate(sub_path):
    path = os.path.join(root_path, path_name)
    # task = [0]*task_num
    # task[i] = 1
    for root, dirs, files in os.walk(path):
        task_nums_all.append(len(files))

# train set for pre-training
random_index = []
task_morph_num = []
for i in range(len(task_nums_all)):
    random_index.append(random.sample(range(task_nums_all[i]),round(task_nums_all[i]*0.9)))
    task_morph_num.append(round(task_nums_all[i]*0.9))
task_morph_num = []
for i in range(len(task_nums_all)):
    #random_index.append(random.sample(range(task_nums_all[i]),round(task_nums_all[i]*0.9)))
    task_morph_num.append(round(task_nums_all[i]*0.9))
for i, path_name in enumerate(sub_path):
    path = os.path.join(root_path, path_name)
    task = [0]*task_num
    task[i] = 1
    for root, dirs, files in os.walk(path):
        #print(i,1)
        #cnt = 0
        for j,file in enumerate(files):
            #cnt+=1
            if j not in random_index[i]:
                #task_morph_num.append(cnt-1)
                continue
            if os.path.splitext(file)[-1] == '.npz':
                data = np.load(os.path.join(root,file))["arr_0"]
                if flag:
                    robos = morph_to_vec(data).reshape(1,25*5)
                    tasks = torch.tensor(task).reshape(1,-1)
                    flag = False
                else:
                    robos = torch.cat((robos, morph_to_vec(data).reshape(1,25*5)), axis=0)
                    tasks = torch.cat((tasks, torch.tensor(task).reshape(1,-1)), axis=0)

### RoboLDA training
model = RoboLDA(task_num=task_num, individual_num=6, organ_num=6, comp_num=10, voxel_num=5, robot_num=robos.shape[0], hidden=128, dropout=0)
ELBO = TraceEnum_ELBO(max_plate_nesting=1, strict_enumeration_warning=False)
guide = config_enumerate(model.guide, expand=True)
optim = Adam({"lr": 0.0001})
#SVI = SVI(model.model, guide, optim, loss = simple_elbo_kl_annealing)
SVI = SVI(model.model, guide, optim, loss = ELBO)
num_epochs = 400

# Inference
logtheta1_loc, logtheta2_loc, logtheta3_loc, logtheta3_scale, logtheta4_loc, logtheta4_scale, inds, organs, organ_logits = model.guide(robos.float(), tasks.float())

# save parameters
torch.save({"logtheta1_loc":logtheta1_loc,
            "logtheta2_loc":logtheta2_loc,
            "logtheta3_loc":logtheta3_loc,
            "logtheta4_loc":logtheta4_loc},"logtheta_loc.pt")

### save result of train set
organ_logits = organ_logits.reshape(-1,25,6)
organs_type = []
for i in range(organ_logits.shape[0]):
    # organ = organs[i,:,:].reshape(25,6)
    # organ_type = torch.nonzero(organ)[:,1]
    # organ_type = organ_type.reshape(1,5,5)
    organ_type = torch.argmax(organ_logits[i,:,:], dim=-1).reshape(1,5,5)
    organs_type.append(organ_type)
organs_type = torch.cat(organs_type)
operator_2 = operator(5).long()
morphs = vec_to_morph(robos,operator_2,5)
task_morph_num.insert(0,0)
task_morph_num_sum = list(accumulate(task_morph_num))

for i in range(len(task_morph_num_sum)-1):
    os.makedirs(save_root_path+sub_path[i])
    task_path = save_root_path+sub_path[i]+'/'
    task_morph = morphs[task_morph_num_sum[i]:task_morph_num_sum[i+1],:,:]
    torch.save(task_morph,task_path+'morph.pt')
    task_organ = organs_type[task_morph_num_sum[i]:task_morph_num_sum[i+1],:,:]
    torch.save(task_organ,task_path+'organ.pt')
    torch.save(organ_logits,task_path+'organ_logits.pt')

### test set
test_flag = True
for i, path_name in enumerate(sub_path):
    path = os.path.join(root_path, path_name)
    print(sub_path[i])
    task = [0]*task_num
    task[i] = 1
    for root, dirs, files in os.walk(path):
        #print(i,1)
        #cnt = task_morph_num[i]
        #new_cnt = 0
        for j,file in enumerate(files):
            #new_cnt+=1
            if j in random_index[i]:
                if j == len(files)-1:
                    _, _, _, _, _, _, _, _, test_organ_logits = model.guide(torch.cat((test_robo_morphs,robos[:(robos.shape[0]-test_robo_morphs.shape[0]),:]),axis=0).float(), torch.cat((test_tasks,tasks[:(tasks.shape[0]-test_tasks.shape[0]),:]),axis=0).float())
                    test_organ_logits = test_organ_logits.reshape(-1,25,6)
                    print(test_robos.shape)
                    #print(cnt)
                    print(len(files))
                    # 测试集保存路径
                    test_root = 'test/'
                    os.makedirs(test_root+sub_path[i])
                    task_test = test_root+sub_path[i]+'/'
                    torch.save(test_robos,task_test+'morph.pt')
                    test_organs_type = []
                    for i in range(test_organ_logits.shape[0]):
                        # organ = test_organ_logits[i,:,:].reshape(25,-1)
                        # organ_type = torch.nonzero(organ)[:,1]
                        # organ_type = organ_type.reshape(1,5,5)
                        organ_type = torch.argmax(test_organ_logits[i,:,:], dim=-1).reshape(1,5,5)
                        test_organs_type.append(organ_type)
                    test_organs_type = torch.cat(test_organs_type)
                    torch.save(test_organs_type[:(test_robos.shape[0]),:,:],task_test+'organ.pt')
                    torch.save(test_organ_logits[:(test_robos.shape[0]),:,:],task_test+'organ_logits.pt')
                    test_flag = True
                continue
            if os.path.splitext(file)[-1] == '.npz':
                data = np.load(os.path.join(root,file))["arr_0"]
                if test_flag:
                    test_robos = torch.tensor(data).reshape(-1,5,5)
                    test_robo_morphs = morph_to_vec(data).reshape(1,25*5)
                    test_tasks = torch.tensor(task).reshape(1,-1)
                    test_flag = False
                else:
                    test_robos = torch.cat((test_robos, torch.tensor(data).reshape(-1,5,5)), axis=0)
                    test_robo_morphs = torch.cat((test_robo_morphs, morph_to_vec(data).reshape(1,25*5)), axis=0)
                    #拼接到足够的机器人数量去进行输入然后取前几个需要的部分
                    
                    test_tasks = torch.cat((test_tasks, torch.tensor(task).reshape(1,-1)), axis=0)
                    #tasks = torch.cat((tasks, torch.tensor(task).reshape(1,-1)), axis=0)
                    
                    if j == len(files)-1:
                        _, _, _, _, _, _, _, _, test_organ_logits = model.guide(torch.cat((test_robo_morphs,robos[:(robos.shape[0]-test_robo_morphs.shape[0]),:]),axis=0).float(), torch.cat((test_tasks,tasks[:(tasks.shape[0]-test_tasks.shape[0]),:]),axis=0).float())
                        test_organ_logits = test_organ_logits.reshape(-1,25,6)
                        print(test_robos.shape)
                        #print(cnt)
                        print(len(files))
                        # 测试集保存root_path
                        test_root = 'test/'
                        os.makedirs(test_root+sub_path[i])
                        task_test = test_root+sub_path[i]+'/'
                        torch.save(test_robos,task_test+'morph.pt')
                        test_organs_type = []
                        for i in range(test_organ_logits.shape[0]):
                            # organ = test_organ_logits[i,:,:].reshape(25,-1)
                            # organ_type = torch.nonzero(organ)[:,1]
                            # organ_type = organ_type.reshape(1,5,5)
                            organ_type = torch.argmax(test_organ_logits[i,:,:], dim=-1).reshape(1,5,5)
                            test_organs_type.append(organ_type)
                        test_organs_type = torch.cat(test_organs_type)
                        torch.save(test_organs_type[:(test_robos.shape[0]),:,:],task_test+'organ.pt')
                        torch.save(test_organ_logits[:(test_robos.shape[0]),:,:],task_test+'organ_logits.pt')
                        test_flag = True

