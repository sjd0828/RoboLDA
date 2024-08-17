# This implementation follows Contextual Modulation. 

import numpy as np, torch, math
from torch import nn
from a2c_ppo_acktr.distributions import FixedNormal

from .Transformer import TransformerEncoder
from .Transformer import TransformerEncoderLayerResidual

class Policy(nn.Module):
    
    def __init__(self, body_size, structure, transformer_mu, transformer_v, global_size, organs):
        # body_size: height and width of robot design, eg. 5
        # structure: voxel matrix
        # transformer: the modular control network shared across robots
        # global_size: dimension of global observation
        
        super(Policy, self).__init__()
        
        self.structure = structure
        self.body_size = body_size
        self.global_size = global_size
        self.organs = organs
        
        self.mu_net = transformer_mu
        self.v_net = transformer_v
        
        self.relu = nn.ReLU()
        self.dist = FixedNormal
        
        actuators = (structure.body.flatten()==3) + (structure.body.flatten()==4)
        self.actuator_mask = []
        for e,v in enumerate(actuators):
            if v:
                actuator_mask_row = [0]*self.body_size**2
                actuator_mask_row[e] = 1
                self.actuator_mask.append(actuator_mask_row)
        self.actuator_mask = torch.tensor(self.actuator_mask, requires_grad=False)
        
        self.is_recurrent = False # unused
        self.recurrent_hidden_state_size = 100 # unused
        
        self.scale = nn.Parameter(torch.rand(np.sum(actuators)), requires_grad=True)
        # self.scale = torch.tensor(0.9, requires_grad=False) # fixed action std
        
    def convert_obs(self, inputs):
        # obtain local and global observations and concat them together
        local_obs = inputs[:, self.global_size:].reshape(inputs.shape[0], 8, self.body_size**2).permute(0,2,1)
        global_obs = inputs[:, 0:self.global_size].unsqueeze(1).repeat(1,self.body_size**2,1)
        return torch.cat((global_obs, local_obs), dim=2)
    
    def get_dist(self, inputs, organs):
        # obtain action distributions for actuators
        
        inputs = self.convert_obs(inputs)
        
        # input structure to generate robot-specific params(, as well as to obtain neighbor masks)
        # obtain means (and variances) for each voxel, no matter actuator or not
        
        action_signals_loc = self.mu_net(inputs, organs, self.structure.body)
        
        # extract action signals of actuators
        action_signals_loc = torch.bmm(self.actuator_mask.to(action_signals_loc.device).float().unsqueeze(0).repeat(inputs.shape[0],1,1), 
                                       action_signals_loc).squeeze(2) # shape: (batch_size, # actuators)
        
        # return self.dist(action_signals_loc, self.scale.unsqueeze(0).repeat(inputs.shape[0], action_signals_loc.shape[1]).to(inputs.device))
        return self.dist(action_signals_loc, self.scale.exp().unsqueeze(0).repeat(inputs.shape[0], 1).to(inputs.device))
    
    def act(self, inputs, organs, deterministic=False, act=True):
        dist = self.get_dist(inputs, organs)
        action_log_probs = None
        if act:
            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()
            action_log_probs = dist.log_probs(action)
        else: # only evaluate state value
            action = None
        
        inputs = self.convert_obs(inputs)
        value = self.v_net(inputs, organs, self.structure.body).squeeze(2)
        #num_voxel = torch.tensor(sum(self.structure.body.flatten()!=0)).to(inputs.device).float()
        #value = torch.sum(torch.tensor(self.structure.body.flatten()!=0).to(inputs.device).float()*value, dim=1)/num_voxel
        value = torch.sum(value, dim=1)/25
        value = value.reshape(-1,1)
        
        rnn_hxs = torch.zeros(100) # unused
        
        return value, action, action_log_probs, rnn_hxs
    
    def get_value(self, inputs, organs):
        value, _, _, _ = self.act(inputs, organs, act=False)
        return value
    
    def evaluate_actions(self, inputs, organs, action):
        value, _, _, _ = self.act(inputs, organs, act=False)
        dist = self.get_dist(inputs, organs)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        rnn_hxs = None
        return value, action_log_probs, dist_entropy, rnn_hxs
        

class ATTBase(nn.Module):
    # model architecture shared between actor and critic
    
    def __init__(self, h_dim, obs_dim, action_dim, hidden_dim, body_size, n_head, dropout):
        # h_dim: dimension of global observation 
        # obs_dim: dimension of observation per voxel
        # action_dim: dimension of action per actuator
        # hidden_dim: # hidden units in transformer
        # robo_emb: dimension of robot design encoding
        # body_size: height and width of robot design
        # n_head: number of heads in attention
        # dropout: probability of drop-out
        
        super(ATTBase, self).__init__()
        
        self.h_dim = h_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.body_size = body_size
        self.n_head = n_head
        self.dropout = nn.Dropout(p=dropout)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        self.encoder = nn.Linear(self.obs_dim, self.hidden_dim)
        
        self.h_encoder1 = nn.Linear(self.h_dim, 64)
        self.h_encoder2 = nn.Linear(64, 64)
        
        self.decoder1 = nn.Linear(self.hidden_dim+64, 64)
        self.decoder2 = nn.Linear(64, self.action_dim)
        
        # intra-organ attention
        #self.attention_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.n_head, dropout=0.1, batch_first=True, dim_feedforward=1024)
        #self.attention = nn.TransformerEncoder(self.attention_layer, num_layers=1)
        attention_layer1 = TransformerEncoderLayerResidual(self.hidden_dim, self.n_head, 1024, 0.1)
        self.transformer_encoder1 = TransformerEncoder(attention_layer1, 1, norm=None)
        
        # fully-connected attention
        #self.attention_layer_2 = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.n_head, dropout=0.1, batch_first=True, dim_feedforward=1024)
        #self.attention_2 = nn.TransformerEncoder(self.attention_layer_2, num_layers=2)
        attention_layer2 = TransformerEncoderLayerResidual(self.hidden_dim, self.n_head, 1024, 0.1)
        self.transformer_encoder2 = TransformerEncoder(attention_layer2, 3, norm=None)
        self.transformer_encoder3 = TransformerEncoder(attention_layer2, 1, norm=None)
        
        self.pos = nn.Parameter(torch.randn(self.body_size**2, self.hidden_dim), requires_grad=True)
        
    def forward(self, inputs, organs, structure):
        
        # compute intra-organ mask
        intra_mask = torch.mm(organs.T, organs)
        intra_mask = intra_mask==0
        
        # local observation encoding
        temp = self.encoder(inputs[:,:,self.h_dim:])
        
        # scale, add position embedding, and dropout
        temp = self.dropout(temp * math.sqrt(self.hidden_dim) + self.pos.unsqueeze(0).repeat(inputs.shape[0],1,1))
        
        # Ablation1 local observation attention (intra-organ)
        # temp = self.transformer_encoder1(temp, mask = intra_mask.to(temp.device))
        # temp = self.transformer_encoder2(temp)
        
        # Ablation2 local observation attention (intra-organ)
        # temp = self.transformer_encoder2(temp)
        # temp = self.transformer_encoder1(temp, mask = intra_mask.to(temp.device))
        # temp = self.transformer_encoder3(temp)
        
        # Ablation3 local observation attention (intra-organ)
        # temp = self.transformer_encoder2(temp)
        # temp = self.transformer_encoder1(temp, mask = intra_mask.to(temp.device))
        # temp = self.transformer_encoder3(temp)
        
        # Ablation4 local observation attention (intra-organ)
        # temp = self.transformer_encoder2(temp)
        # temp = self.transformer_encoder1(temp, mask = intra_mask.to(temp.device))
        # temp = self.transformer_encoder3(temp)
        
        # local observation attention (full connection)
        temp = self.transformer_encoder2(temp)

        # local observation attention (intra-organ) & mask in the 5th layer
        temp = self.transformer_encoder1(temp, mask = intra_mask.to(temp.device))
        
        
        # global observation encoding
        h_temp = self.h_encoder1(inputs[:,:,0:self.h_dim])
        h_temp = self.h_encoder2(self.relu(h_temp))
        temp = torch.concat((h_temp, temp), dim=2)
        
        # decode
        temp = self.relu(self.decoder1(temp))
        temp = self.decoder2(temp)
        
        return temp
