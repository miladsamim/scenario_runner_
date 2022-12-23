import torch
import torch.nn as nn
import torch.nn.functional as F

from . import HDMapSensorDQN, HDMapSensorDQN_V0, PositionalEncoding


class HDDriveDQN_V0_GRU(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.sensor_net = HDMapSensorDQN_V0(args.img_shape, args.ego_dim, args)
        self.temporal_net = nn.GRU(args.h_size*3, 3*args.h_size, num_layers=1)
        self.out = nn.Linear(3*args.h_size, args.simple_space_dim)
        self.residual = args.residual
    
    def forward(self, bev_Xs, front_Xs, acc_Xs, comp_Xs, gyro_Xs, vel_Xs, act_Xs):
        hidden_states = []
        for i, sensor_state in enumerate(zip(bev_Xs, front_Xs, acc_Xs, comp_Xs, gyro_Xs, vel_Xs)):
            # print(list(map(lambda x: x.shape, sensor_state)))
            state_h = self.sensor_net(*sensor_state)
            hidden_states.append(state_h)
        hidden_states = torch.stack(hidden_states, dim=0) # seqLen X batchSize X h_size 
        out, h_n = self.temporal_net(hidden_states)
        h_n = h_n[-1]
        if self.residual:
            h_n = h_n + state_h

        return self.out(F.relu(h_n)) 

class HDDriveDQN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.sensor_net = HDMapSensorDQN(args.img_shape, args.ego_dim, args)
        self.fusion_net = nn.Transformer(d_model=args.h_size, nhead=args.n_head, num_encoder_layers=args.n_encs_f,
                                         num_decoder_layers=args.n_decs_f, dim_feedforward=4*args.h_size, 
                                         dropout=args.t_dropout, norm_first=args.device, device=args.device)
        self.temporal_net = nn.Transformer(d_model=args.h_size, nhead=args.n_head, num_encoder_layers=args.n_encs_t,
                                         num_decoder_layers=args.n_decs_t, dim_feedforward=4*args.h_size, 
                                         dropout=args.t_dropout, norm_first=args.device, device=args.device)
        self.positional_encoder = PositionalEncoding(d_model=args.h_size, max_len=64) # max number of time steps
        
        if args.simple_space:
            self.act_dec_idx = torch.arange(args.simple_space_dim, device=args.device)
            self.act_embedding = nn.Embedding(35, 128, device=args.device)
            self.act_embedding.weight.requires_grad = False 
            self.n_act_nets = args.simple_space_dim
            self.out = nn.Linear(args.h_size, 1)
        else:
            self.steering_net = nn.Sequential(
                nn.Linear(args.h_size, args.act_steering_dim)
            )
            self.throttle_net = nn.Sequential(
                nn.Linear(args.h_size, args.act_throttle_dim)
            )
            self.brake_net = nn.Sequential(
                nn.Linear(args.h_size, args.act_brake_dim)
            )
            self.t_or_b_net = nn.Sequential(
                nn.Linear(args.h_size, 1),
                nn.Sigmoid(),
            )
            self.n_act_nets = 4

    def forward(self, bev_Xs, front_Xs, acc_Xs, comp_Xs, gyro_Xs, vel_Xs, act_Xs):
        n_frames, b_size = bev_Xs.shape[0], bev_Xs.shape[1]
        hidden_states = []
        dec_in = torch.ones(1, b_size, self.args.h_size, requires_grad=False, device=self.args.device)
        for i, sensor_state in enumerate(zip(bev_Xs, front_Xs, acc_Xs, comp_Xs, gyro_Xs, vel_Xs)):
            # print(list(map(lambda x: x.shape, sensor_state)))
            bev_X_h, front_X_h, ego_X_h, vel_X_h = self.sensor_net(*sensor_state)
            act_X_h = self.act_embedding(act_Xs[i]).unsqueeze(0)
            sensor_X = torch.concat([bev_X_h, front_X_h, ego_X_h, vel_X_h, act_X_h], dim=0)
            hidden_state = self.fusion_net(sensor_X, dec_in)
            hidden_states.append(hidden_state)

        hidden_states = torch.concat(hidden_states) # seqLen X batchSize X h_size 
        hidden_states = self.positional_encoder(hidden_states)
        # dec_in = torch.rand(self.n_act_nets, b_size, self.args.h_size, requires_grad=False, device=self.args.device)
        dec_in = self.act_embedding(self.act_dec_idx.repeat(b_size, 1)).transpose(0,1)
        hidden_state = self.temporal_net(hidden_states, dec_in) # seq_len X batchSize X h_size 
        if self.args.simple_space:
            hidden_state = hidden_state.transpose(0, 1) # seq,batch,... -> batch,seq,...
            return self.out(hidden_state).squeeze(2) 
        else :
            steering_q_vals = self.steering_net(hidden_state[0])
            throttle_q_vals = self.steering_net(hidden_state[1])
            brake_q_vals = self.steering_net(hidden_state[2])
            t_or_b_vals = self.t_or_b_net(hidden_state[3])
            return steering_q_vals, throttle_q_vals, brake_q_vals, t_or_b_vals