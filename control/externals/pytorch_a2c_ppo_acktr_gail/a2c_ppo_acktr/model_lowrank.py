import numpy as np
import torch
from torch import nn
from a2c_ppo_acktr.distributions import FixedNormal
import math


class Policy(nn.Module):

    def __init__(self, global_size, body_size, organs, structure, transformer_mu, transformer_v):
        super(Policy, self).__init__()
        self.organs = organs
        self.body_size = body_size
        self.num_organs = (organs.sum(axis=1) != 0).sum().item()
        self.structure = structure
        self.global_size = global_size

        self.mu_net = transformer_mu
        self.v_net = transformer_v

        self.relu = nn.ReLU()
        self.is_recurrent = False
        self.recurrent_hidden_state_size = 10

        actuators = (structure.body.flatten()==3) + (structure.body.flatten()==4)
        self.actuator_mask = []
        for e, v in enumerate(actuators):
            if v:
                actuator_mask_row = [0]*self.body_size**2
                actuator_mask_row[e] = 1
                self.actuator_mask.append(actuator_mask_row)
        self.actuator_mask = torch.tensor(self.actuator_mask)

        self.scale = nn.Parameter(torch.rand(np.sum(actuators)), requires_grad=True)

        self.dist = FixedNormal

    def convert_obs(self, inputs):
        local_obs = inputs[:, self.global_size:].reshape(inputs.shape[0], 8, self.body_size ** 2).permute(0, 2, 1)
        global_obs = inputs[:, 0:self.global_size]
        global_obs = global_obs.unsqueeze(1).repeat(1, self.body_size**2, 1)
        inputs = torch.cat((global_obs.to(local_obs.device), local_obs), 2)
        return inputs

    def get_dist(self, inputs, organs):
        inputs = self.convert_obs(inputs)
        action_signals_loc = self.mu_net(inputs, organs, self.structure.body)
        action_signals_loc = torch.bmm(
            self.actuator_mask.to(action_signals_loc.device).float().unsqueeze(0).repeat(inputs.shape[0], 1, 1),
            action_signals_loc
        ).squeeze(2)
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
        else:
            action = None

        inputs = self.convert_obs(inputs)
        value = self.v_net(inputs, organs, self.structure.body).squeeze(2)
        value = torch.sum(value, dim=1) / 25
        value = value.reshape(-1, 1)

        rnn_hxs = torch.zeros(100)

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

    def __init__(self, obs_dim, h_dim, action_dim, hidden_dim, body_size, n_head, dropout):
        super(ATTBase, self).__init__()

        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.h_dim = h_dim
        self.action_dim = action_dim
        self.body_size = body_size

        self.relu = nn.ReLU()

        self.pos = nn.Parameter(torch.Tensor(self.body_size**2, self.hidden_dim), requires_grad=True)
        nn.init.normal_(self.pos)
        self.encoder_pos_1 = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_pos_2 = nn.Linear(hidden_dim, hidden_dim)

        self.encoder = nn.Linear(in_features=self.obs_dim, out_features=self.hidden_dim)
        self.h_encoder = nn.Linear(self.h_dim, 8)

        # intra-synergy attention
        self.intra_attention = nn.MultiheadAttention(self.hidden_dim, n_head, dropout=dropout, batch_first=True)
        self.intra_norm1 = nn.LayerNorm(self.hidden_dim)
        self.intra_dropout0 = nn.Dropout(dropout)
        self.intra_linear1 = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        self.intra_dropout1 = nn.Dropout(dropout)
        self.intra_linear2 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.intra_dropout2 = nn.Dropout(dropout)
        self.intra_norm2 = nn.LayerNorm(self.hidden_dim)

        # inter-synergy attention (layer 1)
        self.inter_attention_1 = nn.MultiheadAttention(self.hidden_dim, n_head, dropout=dropout, batch_first=True)
        self.inter_norm11 = nn.LayerNorm(self.hidden_dim)
        self.inter_dropout10 = nn.Dropout(dropout)
        self.inter_linear11 = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        self.inter_dropout11 = nn.Dropout(dropout)
        self.inter_linear12 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.inter_dropout12 = nn.Dropout(dropout)
        self.inter_norm12 = nn.LayerNorm(self.hidden_dim)

        # inter-synergy attention (layer 2)
        self.inter_attention_2 = nn.MultiheadAttention(self.hidden_dim, n_head, dropout=dropout, batch_first=True)
        self.inter_norm21 = nn.LayerNorm(self.hidden_dim)
        self.inter_dropout20 = nn.Dropout(dropout)
        self.inter_linear21 = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        self.inter_dropout21 = nn.Dropout(dropout)
        self.inter_linear22 = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.inter_dropout22 = nn.Dropout(dropout)
        self.inter_norm22 = nn.LayerNorm(self.hidden_dim)

        self.h_encoder1 = nn.Linear(self.h_dim, 64)
        self.h_encoder2 = nn.Linear(64, 64)

        self.decoder1 = nn.Linear(self.hidden_dim + 64, 64)
        self.decoder2 = nn.Linear(64, self.action_dim)

    def forward(self, inputs, organs, structure):
        # intra-synergy mask: voxels only attend within the same organ
        intra_mask = torch.mm(organs.T, organs)
        intra_mask = intra_mask == 0

        # encode local observations, add positional embedding
        encoded_obs = self.relu(self.encoder(inputs[:, :, self.h_dim:])) * math.sqrt(self.hidden_dim)
        encoded_obs = encoded_obs + self.pos

        # intra-synergy attention
        intra_obs, _ = self.intra_attention(
            encoded_obs, encoded_obs, encoded_obs,
            attn_mask=intra_mask.to(encoded_obs.device)
        )
        intra_obs = self.intra_dropout0(intra_obs) + encoded_obs
        intra_obs = self.intra_norm1(intra_obs)
        intra_obs_temp = self.intra_linear2(self.intra_dropout1(self.relu(self.intra_linear1(intra_obs))))
        intra_obs = self.intra_dropout2(intra_obs_temp) + intra_obs
        intra_obs = self.intra_norm2(intra_obs)

        # aggregate voxel representations into organ representations
        organs = organs.reshape(1, organs.shape[0], organs.shape[1]).float()
        organs = torch.repeat_interleave(organs, inputs.shape[0], dim=0)
        denom = organs.to(intra_obs.device).sum(axis=2, keepdims=True)
        denom[denom == 0] = 1
        organ_obs = torch.bmm(organs.to(intra_obs.device), intra_obs) / denom.float()

        # inter-synergy attention (layer 1)
        inter_obs, _ = self.inter_attention_1(organ_obs, organ_obs, organ_obs)
        inter_obs = organ_obs + self.inter_dropout10(inter_obs)
        inter_obs = self.inter_norm11(inter_obs)
        inter_obs_temp = self.inter_linear12(self.inter_dropout11(self.relu(self.inter_linear11(inter_obs))))
        inter_obs = inter_obs + self.inter_dropout12(inter_obs_temp)
        inter_obs = self.inter_norm12(inter_obs)

        # inter-synergy attention (layer 2)
        inter_obs2, _ = self.inter_attention_2(inter_obs, inter_obs, inter_obs)
        inter_obs = inter_obs + self.inter_dropout20(inter_obs2)
        inter_obs = self.inter_norm21(inter_obs)
        inter_obs_temp = self.inter_linear22(self.inter_dropout21(self.relu(self.inter_linear21(inter_obs))))
        inter_obs = inter_obs + self.inter_dropout22(inter_obs_temp)
        inter_obs = self.inter_norm22(inter_obs)
        organ_signals = inter_obs

        # distribute organ signals back to voxels via learned position weights
        pos = self.relu(self.encoder_pos_1(self.pos))
        pos = self.encoder_pos_2(pos)

        signal_weights_temp = torch.mm(pos, pos.T)
        signal_weights_temp = signal_weights_temp - torch.diag_embed(torch.diag(signal_weights_temp))

        organs = organs[0]
        denom = torch.mm((signal_weights_temp != 0).float(), organs.T.to(signal_weights_temp.device))
        denom[denom == 0] = 1
        signal_weights = torch.mm(signal_weights_temp, organs.T.to(signal_weights_temp.device)) / denom.float()

        signal_weights = signal_weights.reshape(1, signal_weights.shape[0], signal_weights.shape[1])
        signal_weights = torch.repeat_interleave(signal_weights, inputs.shape[0], dim=0)
        voxel_signals = torch.bmm(signal_weights, organ_signals)

        # encode global observations
        h_temp = self.h_encoder1(inputs[:, :, 0:self.h_dim])
        h_temp = self.h_encoder2(self.relu(h_temp))

        temp = torch.concat((h_temp, voxel_signals), dim=2)

        # decode
        temp = self.relu(self.decoder1(temp))
        temp = self.decoder2(temp)

        return temp
