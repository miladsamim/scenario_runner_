import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import PositionalEncoding

class HDMapSensorDQN_V0(nn.Module):
    def __init__(self, in_shape, ego_dim, args, p=0.5):
        super().__init__()
        in_channels = in_shape[0]
        self.bev_cnn = Nature_Paper_Conv_Dropout_Torch(in_channels, args.h_size, 
                                                       image_shape=(in_shape[1], in_shape[2]), p=p)                                                       
        self.front_cnn = Nature_Paper_Conv_Dropout_Torch(in_channels, args.h_size, 
                                                       image_shape=(in_shape[1], in_shape[2]), p=p)
        self.ego_dense = nn.Sequential(
                            nn.Linear(ego_dim+1, 512),
                            nn.ReLU(),
                            nn.Linear(512, args.h_size),
                            nn.ReLU(),
                        )
        self.vel_norm_speed = 130 #km/h

    def _cnn_out_dim(self, input_shape):
        X = torch.rand(1, *input_shape)
        return self.bev_cnn(X).shape[1]

    def forward(self, bev_X, front_X, acc_X, comp_X, gyro_X, vel_X):
        bev_X = self.bev_cnn(bev_X)
        front_X = self.front_cnn(front_X)
        vel_X = (vel_X / self.vel_norm_speed).unsqueeze(-1).type(torch.float32)
        ego_X = self.ego_dense(torch.concat([acc_X, comp_X, gyro_X, vel_X], dim=1))
        state_h = torch.cat([bev_X, front_X, ego_X], dim=1)
        return state_h

class HDMapSensorDQN(nn.Module):
    def __init__(self, in_shape, ego_dim, args, h_size=128, p=0.5):
        super().__init__()
        in_channels = in_shape[0]
        self.bev_cnn = build_cnn(in_channels, p=p)
        # Assume images are 3X96X96
        assert self._cnn_out_dim(in_shape) == 2304
        self.bev_dense = build_cnn_to_dense(2304, args.h_size)
        self.front_cnn = build_cnn(in_channels, p=p)
        self.front_dense = build_cnn_to_dense(2304, args.h_size)
        self.ego_dense = nn.Sequential(
                            nn.Linear(ego_dim, 64),
                            nn.ReLU(),
                            nn.Linear(64, args.h_size),
                            nn.ReLU(),
                        )
        self.emb = nn.Embedding(1, h_size, device=args.device)
        self.emb.weight.requires_grad = False

        # Assume max speed 500 km/h
        self.positional_encoder = PositionalEncoding(h_size, max_len=500)

    def _cnn_out_dim(self, input_shape):
        X = torch.rand(1, *input_shape)
        return self.bev_cnn(X).shape[1]

    def forward(self, bev_X, front_X, acc_X, comp_X, gyro_X, vel_X):
        bev_X = self.bev_dense(self.bev_cnn(bev_X)).unsqueeze(dim=0)
        front_X = self.front_dense(self.front_cnn(front_X)).unsqueeze(dim=0)
        ego_X = self.ego_dense(torch.concat([acc_X, comp_X, gyro_X], dim=1)).unsqueeze(dim=0)
        vel_X_temp = self.positional_encoder.pe[vel_X.to(torch.long)].transpose(0,1)
        vel_X = vel_X_temp + self.emb(torch.zeros_like(vel_X).long())
        return bev_X, front_X, ego_X, vel_X

def GetPadConfig(input_size, pad_mode, pad_left, pad_right, kernel, stride):
    """Calculates the padding config.
    Inputs:
        input_size (int): the input image size.
        pad_mode (string): either 'SAME', or 'EXPLICIT'
        pad_left, pad_right (int): the padding values used in 'EXPLICIT' mode. Does not have effect in 'SAME'.
        kernel: the kernel size.
        stride: the stride size.
    Returns:
        output_size (int): the output image size.
        pad_left, pad_right (int): the padding values on the left and right.
    """
    if (pad_mode == 'SAME'):
        output_size = int(math.ceil(float(input_size) / float(stride)))
        pad_total = int((output_size - 1) * stride + kernel - input_size)
        pad_left = int(pad_total / 2)
        pad_right = pad_total - pad_left
    elif (pad_mode == 'EXPLICIT'):
        output_size = int((input_size + pad_left + pad_right - kernel) / stride) + 1
    else:
        raise RuntimeError('Wrong pad mode.')
    return output_size, pad_left, pad_right


class Nature_Paper_Conv_Dropout_Torch(nn.Module):
    def __init__(self, in_channels, action_size, p=0.5, image_shape=(96,96)):
        super().__init__()
        self.p=p
        pads = self._compute_paddings()
        self.layer1_cnn = nn.Conv2d(in_channels, out_channels=32, kernel_size=(8,8), stride=(4,4), 
                                padding=(pads[0][0],pads[0][1]))
        self.layer2_cnn = nn.Conv2d(32, out_channels=64, kernel_size=(4,4), stride=(2,2), 
                                padding=(pads[1][0],pads[1][1]))
        self.layer3_cnn = nn.Conv2d(64, out_channels=64, kernel_size=(3,3), stride=(1,1), 
                                padding='same')
        assert self._get_dense_input_neurons(in_channels, image_shape) == 9216 # 3,96,96
        self.layer_4_dense = nn.Linear(9216, 512)
        self.out_layer = nn.Linear(512, action_size)

    def init_weights(m):
        if type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    def forward(self, X):
        X = F.relu(self.layer1_cnn(X))
        X = F.dropout(F.relu(self.layer2_cnn(X)), self.p)
        X = F.relu(self.layer3_cnn(X))
        X = torch.flatten(X, start_dim=1)
        X = F.relu(self.layer_4_dense(X))
        return self.out_layer(X)

    def _get_dense_input_neurons(self, in_channels, image_shape):
        X = torch.rand(1, in_channels, *image_shape)
        return torch.flatten(self.layer3_cnn(self.layer2_cnn(self.layer1_cnn(X))), start_dim=1).shape[1]
    
    def _compute_paddings(self):
        """Required as torch prohibits padding='same' when stride > 1, hence we use 
           below to get same effect as tensorflow"""
        _, pad_left_layer1, pad_right_layer2 = GetPadConfig(96, 'SAME', 0,0,8,4)
        _, pad_left_layer2, pad_right_layer2 = GetPadConfig(96, 'SAME', 0,0,4,2)
        return [[pad_left_layer1, pad_left_layer1],[pad_left_layer2, pad_right_layer2]]

def build_cnn(in_channels, p=0.5):
    """Made for image sizes of 96X96"""
    _, pad_left_layer1, pad_right_layer2 = GetPadConfig(96, 'SAME', 0,0,8,4)
    _, pad_left_layer2, pad_right_layer2 = GetPadConfig(96, 'SAME', 0,0,4,2)
    pads = [[pad_left_layer1, pad_left_layer1],[pad_left_layer2, pad_right_layer2]]
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels=32, kernel_size=(8,8), stride=(4,4), padding=(pads[0][0],pads[0][1])),
        nn.ReLU(),
        nn.Dropout(p),
        nn.Conv2d(32, out_channels=64, kernel_size=(4,4), stride=(2,2), padding=(pads[1][0],pads[1][1])),
        nn.ReLU(),
        nn.Conv2d(64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding='same'),
        nn.MaxPool2d((2,2)),
        nn.ReLU(),
        nn.Flatten(),
    )

def build_cnn_to_dense(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
    )