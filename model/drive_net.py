import torch
import torch.nn as nn
import torch.nn.functional as F

from . import HDMapSensorDQN, PositionalEncoding

class HDDriveDQN(nn.Module):
    def __init__(self, args):
        super(HDDriveDQN, self).__init__()
        self.args = args
        self.sensor_net = HDMapSensorDQN(args.img_shape, args.ego_dim)
        self.fusion_net = nn.Transformer(d_model=args.h_size, nhead=args.n_head, num_encoder_layers=args.n_encs_f,
                                         num_decoder_layers=args.n_decs_f, dim_feedforward=4*args.h_size, 
                                         dropout=args.t_dropout, norm_first=args.device, device=args.device)
        self.temporal_net = nn.Transformer(d_model=args.h_size, nhead=args.n_head, num_encoder_layers=args.n_encs_t,
                                         num_decoder_layers=args.n_decs_t, dim_feedforward=4*args.h_size, 
                                         dropout=args.t_dropout, norm_first=args.device, device=args.device)
        self.positional_encoder = PositionalEncoding(d_model=args.h_size, max_len=64) # max number of time steps
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
            nn.Linear(args.h_size, 2)
        )
        self.n_act_nets = 4

    def forward(self, bev_Xs, front_Xs, acc_Xs, comp_Xs, gyro_Xs, vel_Xs):
        n_frames, b_size = bev_Xs.shape[0], bev_Xs.shape[1]
        hidden_states = []
        dec_in = torch.zeros(1, b_size, self.args.h_size, requires_grad=False, device=self.args.device)
        for sensor_state in zip(bev_Xs, front_Xs, acc_Xs, comp_Xs, gyro_Xs, vel_Xs):
            # print(list(map(lambda x: x.shape, sensor_state)))
            bev_X_h, front_X_h, ego_X_h, vel_X_h = self.sensor_net(*sensor_state)
            sensor_X = torch.concat([bev_X_h, front_X_h, ego_X_h, vel_X_h], dim=0)
            hidden_state = self.fusion_net(sensor_X, dec_in)
            hidden_states.append(hidden_state)

        hidden_states = torch.concat(hidden_states) # seqLen X batchSize X h_size 
        dec_in = torch.zeros(self.n_act_nets, b_size, self.args.h_size, requires_grad=False, device=self.args.device)
        hidden_state = self.temporal_net(hidden_states, dec_in).squeeze(0) # batchSize X h_size 
        steering_q_vals = self.steering_net(hidden_state[0])
        throttle_q_vals = self.steering_net(hidden_state[1])
        brake_q_vals = self.steering_net(hidden_state[2])
        t_or_b_vals = self.t_or_b_net(hidden_state[3])
        return steering_q_vals, throttle_q_vals, brake_q_vals, t_or_b_vals