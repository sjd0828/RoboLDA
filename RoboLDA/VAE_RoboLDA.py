import torch.nn as nn
import torch.nn.functional as F
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.optim import ClippedAdam, Adam
import pyro.distributions as dist
from tqdm import trange

class RoboLDA(nn.Module):
    def __init__(self, task_num, individual_num, organ_num, comp_num, voxel_num, robot_num, hidden, dropout):
        super().__init__()
        
        self.task_num = task_num
        self.individual_num = individual_num
        self.organ_num = organ_num
        self.voxel_num = voxel_num
        self.robot_num = robot_num
        self.activation = F.tanh
        self.hidden = hidden
        self.comp_num=comp_num

        self.voxel_location = torch.zeros(25, 10)
        for r in range(5):
            for c in range(5):
                index = r*5+c
                self.voxel_location[index,r] = 1
                self.voxel_location[index,5+c] = 1
        
        #self.drop = nn.Dropout(dropout)  # to avoid component collapse
        self.drop = nn.Identity()
        
        #### inference of the task, individual and organ of each robot
        
        # voxel -> task
        self.task_fc1 = nn.Linear(voxel_num*25, hidden)
        self.task_fc2 = nn.Linear(hidden, hidden)
        self.task_fc3 = nn.Linear(hidden, task_num)
        self.task_bn = nn.BatchNorm1d(self.task_num, affine=False)
        
        # voxel+task -> individual
        self.ind_fc1 = nn.Linear(voxel_num*25+task_num+task_num*individual_num, hidden)
        self.ind_fc2 = nn.Linear(hidden, hidden)
        self.ind_fc3 = nn.Linear(hidden, individual_num)
        self.ind_bn = nn.BatchNorm1d(self.individual_num, affine=False)
        
        # voxel+task -> organ
        self.organ_fc1 = nn.Linear(voxel_num*25+individual_num+comp_num*individual_num*organ_num, hidden)
        self.organ_fc2 = nn.Linear(hidden, hidden)
        self.organ_fc3 = nn.Linear(hidden, organ_num*25)
        self.organ_bn = nn.BatchNorm1d(self.organ_num*25, affine=False)
        
        #### inference of log_theta
        
        # robots -> log_theta1
        self.logtheta1_fc1 = nn.Linear((task_num+voxel_num*25)*robot_num, hidden)
        self.logtheta1_fc2 = nn.Linear(hidden, hidden)
        self.logtheta1_loc = nn.Linear(hidden, task_num)
        self.logtheta1_scale = nn.Linear(hidden, task_num)
        
        # robots -> log_theta2
        self.logtheta2_fc1 = nn.Linear((task_num+voxel_num*25)*robot_num, hidden)
        self.logtheta2_fc2 = nn.Linear(hidden, hidden)
        self.logtheta2_loc = nn.Linear(hidden, task_num*individual_num)
        self.logtheta2_scale = nn.Linear(hidden, task_num*individual_num)
        
        # robots -> log_theta3
        self.logtheta3_fc1 = nn.Linear((task_num+voxel_num*25)*robot_num, hidden)
        self.logtheta3_fc2 = nn.Linear(hidden, hidden)
        self.logtheta3_loc = nn.Linear(hidden, individual_num*organ_num*comp_num)
        self.logtheta3_scale = nn.Linear(hidden, individual_num*organ_num*comp_num)        
        
        self.organ_weight_1 = nn.Linear(10+individual_num, hidden)
        self.organ_weight_2 = nn.Linear(hidden, hidden)
        self.organ_weight_3 = nn.Linear(hidden, comp_num)
        
        # robots -> log_theta4
        self.logtheta4_fc1 = nn.Linear((task_num+voxel_num*25)*robot_num, hidden)
        self.logtheta4_fc2 = nn.Linear(hidden, hidden)
        self.logtheta4_loc = nn.Linear(hidden, organ_num*voxel_num*25)
        self.logtheta4_scale = nn.Linear(hidden, organ_num*voxel_num*25)
    

    
    @config_enumerate(default="parallel")
    def model(self, robos, tasks, logtheta1=None, logtheta2=None, logtheta3=None, logtheta4=None):
        pyro.module("ss_vae", self)
        
        #pyro.module("decoder", self.decoder)
        # sample log_theta (1+y+n+m)

        log_theta_1 = pyro.sample("logtheta1", dist.Normal(torch.zeros(self.task_num),torch.ones(self.task_num)).to_event(1), obs=logtheta1)
        log_theta_1 = self.drop(F.softmax(log_theta_1.reshape(1, self.task_num),-1))
        
        log_theta_2 = pyro.sample("logtheta2", dist.Normal(torch.zeros(self.task_num*self.individual_num),torch.ones(self.task_num*self.individual_num)).to_event(1), obs=logtheta2)
        log_theta_2 = self.drop(F.softmax(log_theta_2.reshape(self.task_num, self.individual_num),-1))
        
        log_theta_3 = pyro.sample("logtheta3", dist.Normal(torch.zeros(self.individual_num*self.organ_num*self.comp_num),
                                                           torch.ones(self.individual_num*self.organ_num*self.comp_num)).to_event(1), obs=logtheta3)
        log_theta_3 = self.drop(F.softmax(log_theta_3.reshape(self.comp_num, self.individual_num, self.organ_num),-1))
        
        
        log_theta_4 = pyro.sample("logtheta4", dist.Normal(torch.zeros(self.organ_num*self.voxel_num*25), 
                                                           torch.ones(self.organ_num*self.voxel_num*25)).to_event(1), obs=logtheta4)
        log_theta_4 = self.drop(F.softmax(log_theta_4.reshape(25, self.organ_num, self.voxel_num),-1))
        
        n = tasks.shape[0]
        
        with pyro.plate("robots"):
            
            # sample task -> tasks (robos.shape[0], y) 
            task_params = log_theta_1.repeat(n,1)
            tasks = pyro.sample("tasks", dist.OneHotCategorical(probs=task_params).to_event(1), obs=tasks)

            # sample type  # tasks*log_theta_2, sample 
            individual_params = torch.mm(tasks,log_theta_2)
            individuals = pyro.sample("individuals", dist.OneHotCategorical(probs=individual_params).to_event(1))
            
            temp = torch.cat([individuals.unsqueeze(1).repeat(1,25,1), self.voxel_location.unsqueeze(0).repeat(n,1,1)], dim=2) # (n,25,16)
            organ_weight = self.activation(self.organ_weight_1(temp))
            organ_weight = self.activation(self.organ_weight_2(organ_weight))
            organ_weight = self.organ_weight_3(organ_weight) # (n,25,10)
            temp = log_theta_3.reshape(self.comp_num, -1).unsqueeze(0).repeat(n,1,1) # (n,10,6*6)
            temp = torch.bmm(organ_weight, temp).reshape(n,25,self.individual_num,self.organ_num) # (n,25,6,6)
            temp = torch.bmm(individuals.unsqueeze(1).repeat(1,25,1).unsqueeze(2).reshape(n*25,1,self.individual_num), temp.reshape(n*25,self.individual_num,self.organ_num))
            temp = temp.squeeze(1).reshape(n,25,self.organ_num) # (n,25,6)
            organs = pyro.sample("organs",dist.OneHotCategorical(logits=temp).to_event(2))
            
            # sample voxel 
            voxel_params = torch.bmm(organs.permute(1,0,2), log_theta_4).permute(1,0,2)
            # voxel_params = torch.mm(organs.reshape(n*25,self.organ_num),log_theta_4).reshape(n,25,5)
            if robos != None:
                voxels = pyro.sample("voxels",dist.OneHotCategorical(probs=voxel_params).to_event(2), obs=robos.reshape(-1,25,5))
            else:
                voxels = pyro.sample("voxels",dist.OneHotCategorical(probs=voxel_params).to_event(2), obs=robos)
        
        return voxels.reshape(-1,125)

    @config_enumerate(default="parallel")
    def guide(self, robos, tasks):
        #pyro.module("encoder", self.encoder)
        comp_num = self.comp_num
        
        #### inference of logthetas
        with pyro.plate("data"):
            logtheta1_h = self.activation(self.logtheta1_fc1(torch.cat((robos, tasks), axis=1).reshape(1,-1)))
            logtheta1_h = self.activation(self.logtheta1_fc2(logtheta1_h))
            logtheta1_h = self.drop(logtheta1_h)
            logtheta1_loc = self.logtheta1_loc(logtheta1_h).reshape(-1)
            logtheta1_scale = self.logtheta1_scale(logtheta1_h).reshape(-1)
            logtheta1 = pyro.sample("logtheta1", dist.Normal(logtheta1_loc, (0.5*logtheta1_scale).exp()).to_event(1))
        
            logtheta2_h = self.activation(self.logtheta2_fc1(torch.cat((robos, tasks), axis=1).reshape(1,-1)))
            logtheta2_h = self.activation(self.logtheta2_fc2(logtheta2_h))
            logtheta2_h = self.drop(logtheta2_h)
            logtheta2_loc = self.logtheta2_loc(logtheta2_h).reshape(-1)
            logtheta2_scale = self.logtheta2_scale(logtheta2_h).reshape(-1)
            logtheta2 = pyro.sample("logtheta2", dist.Normal(logtheta2_loc, 
                                                         (0.5*logtheta2_scale).exp()).to_event(1))
        
            logtheta3_h = self.activation(self.logtheta3_fc1(torch.cat((robos, tasks), axis=1).reshape(1,-1)))
            logtheta3_h = self.activation(self.logtheta3_fc2(logtheta3_h))
            logtheta3_h = self.drop(logtheta3_h)
            logtheta3_loc = self.logtheta3_loc(logtheta3_h).reshape(-1)
            logtheta3_scale = self.logtheta3_scale(logtheta3_h).reshape(-1)
            logtheta3 = pyro.sample("logtheta3", dist.Normal(logtheta3_loc, 
                                                         (0.5*logtheta3_scale).exp()).to_event(1))
            
            logtheta4_h = self.activation(self.logtheta4_fc1(torch.cat((robos, tasks), axis=1).reshape(1,-1)))
            logtheta4_h = self.activation(self.logtheta4_fc2(logtheta4_h))
            logtheta4_h = self.drop(logtheta4_h)
            logtheta4_loc = self.logtheta4_loc(logtheta4_h).reshape(-1)
            logtheta4_scale = self.logtheta4_scale(logtheta4_h).reshape(-1)
            logtheta4 = pyro.sample("logtheta4", dist.Normal(logtheta4_loc, 
                                                         (0.5*logtheta4_scale).exp()).to_event(1))
        
        #### inference of task, individual and organ of each robot
        with pyro.plate("robos"): 
            # inference of task not used, since task is given
            task_h = self.activation(self.task_fc1(robos))
            task_h = self.activation(self.task_fc2(task_h))
            task_h = self.drop(task_h)
            task_logits = self.task_bn(self.task_fc3(task_h))
            tasks = pyro.sample("tasks", dist.OneHotCategorical(logits=task_logits).to_event(1), obs=tasks)
            
            ind_h = self.activation(self.ind_fc1(torch.cat((robos, tasks, logtheta2.unsqueeze(0).repeat(robos.shape[0],1)), dim=1)))
            ind_h = self.activation(self.ind_fc2(ind_h))
            ind_h = self.drop(ind_h)
            ind_logits = self.ind_bn(self.ind_fc3(ind_h))
            inds = pyro.sample("individuals", dist.OneHotCategorical(logits=ind_logits).to_event(1))
            
            organ_h = self.activation(self.organ_fc1(torch.cat((robos, inds, logtheta3.unsqueeze(0).repeat(robos.shape[0],1)), dim=1)))
            organ_h = self.activation(self.organ_fc2(organ_h))
            organ_h = self.drop(organ_h)
            organ_logits = self.organ_bn(self.organ_fc3(organ_h))
            organs = pyro.sample("organs", dist.OneHotCategorical(logits=organ_logits.reshape(-1,25,6)).to_event(2))
        
        return logtheta1_loc, logtheta2_loc, logtheta3_loc, logtheta3_scale, logtheta4_loc, logtheta4_scale, inds, organs, organ_logits
        
        
    def reset(self, robot_num):
        
        task_num = self.task_num
        individual_num = self.individual_num
        organ_num = self.organ_num
        voxel_num = self.voxel_num
        hidden = self.hidden
        comp_num = self.comp_num
        
        #### inference of log_theta
        
       # robots -> log_theta1
        self.logtheta1_fc1 = nn.Linear((task_num+voxel_num*25)*robot_num, hidden)
        self.logtheta1_fc2 = nn.Linear(hidden, hidden)
        self.logtheta1_loc = nn.Linear(hidden, task_num)
        self.logtheta1_scale = nn.Linear(hidden, task_num)
        
        # robots -> log_theta2
        self.logtheta2_fc1 = nn.Linear((task_num+voxel_num*25)*robot_num, hidden)
        self.logtheta2_fc2 = nn.Linear(hidden, hidden)
        self.logtheta2_loc = nn.Linear(hidden, task_num*individual_num)
        self.logtheta2_scale = nn.Linear(hidden, task_num*individual_num)
        
        # robots -> log_theta3
        self.logtheta3_fc1 = nn.Linear((task_num+voxel_num*25)*robot_num, hidden)
        self.logtheta3_fc2 = nn.Linear(hidden, hidden)
        self.logtheta3_loc = nn.Linear(hidden, individual_num*organ_num*comp_num)
        self.logtheta3_scale = nn.Linear(hidden, individual_num*organ_num*comp_num)        
        
        # robots -> log_theta4
        self.logtheta4_fc1 = nn.Linear((task_num+voxel_num*25)*robot_num, hidden)
        self.logtheta4_fc2 = nn.Linear(hidden, hidden)
        self.logtheta4_loc = nn.Linear(hidden, organ_num*voxel_num*25)
        self.logtheta4_scale = nn.Linear(hidden, organ_num*voxel_num*25)
        
        print("VAE reset. ")

   

