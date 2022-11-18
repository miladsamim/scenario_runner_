import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import PositionalEncoding

class HDMapSensorDQN(nn.Module):
    def __init__(self, in_shape, ego_dim, h_size=128, p=0.5):
        super(HDMapSensorDQN, self).__init__()
        in_channels = in_shape[0]
        self.bev_cnn = build_cnn(in_channels, p=p)
        # Assume images are 3X96X96
        assert self._cnn_out_dim(in_shape) == 2304
        self.bev_dense = build_cnn_to_dense(2304, h_size)
        self.front_cnn = build_cnn(in_channels, p=p)
        self.front_dense = build_cnn_to_dense(2304, h_size)
        self.ego_dense = nn.Sequential(
                            nn.Linear(ego_dim, 64),
                            nn.ReLU(),
                            nn.Linear(64, h_size),
                            nn.ReLU(),
                        )

        # Assume max speed 500 km/h
        self.positional_encoder = PositionalEncoding(h_size, max_len=500)

    def _cnn_out_dim(self, input_shape):
        X = torch.rand(1, *input_shape)
        return self.bev_cnn(X).shape[1]

    def forward(self, bev_X, front_X, acc_X, comp_X, gyro_X, vel_X):
        bev_X = self.bev_dense(self.bev_cnn(bev_X)).unsqueeze(dim=0)
        front_X = self.bev_dense(self.bev_cnn(front_X)).unsqueeze(dim=0)
        ego_X = self.ego_dense(torch.concat([acc_X, comp_X, gyro_X], dim=1)).unsqueeze(dim=0)
        vel_X = self.positional_encoder.pe[vel_X.to(torch.long)].transpose(0,1)
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