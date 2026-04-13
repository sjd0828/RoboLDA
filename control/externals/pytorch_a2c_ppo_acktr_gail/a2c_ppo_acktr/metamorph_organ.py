import numpy as np, torch, math
from torch import nn
from a2c_ppo_acktr.distributions import FixedNormal

from .Transformer import TransformerEncoder
from .Transformer import TransformerEncoderLayerResidual


class Policy(nn.Module):

    def __init__(self, body_size, structure, transformer_mu, transformer_v, global_size, organs):
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
        for e, v in enumerate(actuators):
            if v:
                actuator_mask_row = [0]*self.body_size**2
                actuator_mask_row[e] = 1
                self.actuator_mask.append(actuator_mask_row)
        self.actuator_mask = torch.tensor(self.actuator_mask, requires_grad=False)

        self.is_recurrent = False
        self.recurrent_hidden_state_size = 100

        self.scale = nn.Parameter(torch.rand(np.sum(actuators)), requires_grad=True)

    def convert_obs(self, inputs):
        local_obs = inputs[:, self.global_size:].reshape(inputs.shape[0], 8, self.body_size**2).permute(0, 2, 1)
        global_obs = inputs[:, 0:self.global_size].unsqueeze(1).repeat(1, self.body_size**2, 1)
        return torch.cat((global_obs, local_obs), dim=2)

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

    def __init__(self, h_dim, obs_dim, action_dim, hidden_dim, body_size, n_head, dropout):
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

        self.decoder1 = nn.Linear(self.hidden_dim + 64, 64)
        self.decoder2 = nn.Linear(64, self.action_dim)

        attention_layer1 = TransformerEncoderLayerResidual(self.hidden_dim, self.n_head, 1024, 0.1)
        self.transformer_encoder1 = TransformerEncoder(attention_layer1, 1, norm=None)

        attention_layer2 = TransformerEncoderLayerResidual(self.hidden_dim, self.n_head, 1024, 0.1)
        self.transformer_encoder2 = TransformerEncoder(attention_layer2, 3, norm=None)

        self.pos = nn.Parameter(torch.randn(self.body_size**2, self.hidden_dim), requires_grad=True)

    def forward(self, inputs, organs, structure):
        intra_mask = torch.mm(organs.T, organs)
        intra_mask = intra_mask == 0

        temp = self.encoder(inputs[:, :, self.h_dim:])
        temp = self.dropout(temp * math.sqrt(self.hidden_dim) + self.pos.unsqueeze(0).repeat(inputs.shape[0], 1, 1))

        temp = self.transformer_encoder2(temp)
        temp = self.transformer_encoder1(temp, mask=intra_mask.to(temp.device))

        h_temp = self.h_encoder1(inputs[:, :, 0:self.h_dim])
        h_temp = self.h_encoder2(self.relu(h_temp))
        temp = torch.concat((h_temp, temp), dim=2)

        temp = self.relu(self.decoder1(temp))
        temp = self.decoder2(temp)

        return temp
