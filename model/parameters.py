import torch 
from srunner.tools import dotdict
hd_net_args = dotdict()
hd_net_args.h_size = 128 # dimensionality of latent vectors
hd_net_args.n_head = 8 # number of attention heads
hd_net_args.n_encs_f = 2 # number of encoder layers in the fusion transformer
hd_net_args.n_decs_f = 1 # number of decoder layers in the fusion transformer
hd_net_args.n_encs_t = 2 # number of encoder layers in the temporal transformer
hd_net_args.n_decs_t = 1 # number of decoder layers in the temporal transformer
hd_net_args.norm_first = True # whether to normalize before applying attention layers
hd_net_args.t_dropout = 0.1 # dropout rate in the transformers
hd_net_args.c_dropout = 0.3 # dropout rate in sensor cnns
hd_net_args.device = 'cuda' if torch.cuda.is_available() else 'cpu' # tensor compute device
hd_net_args.img_shape = (3,96,96) # input shape of image based sensors (3X96X96)
hd_net_args.ego_dim = 7 # dimensionality of ego state information
hd_net_args.n_frames = 4
hd_net_args.act_steering_dim = 11 # number of discrete steering actions
hd_net_args.act_throttle_dim = 11 # number of discrete throttle actions 
hd_net_args.act_brake_dim = 11 # number of discrete brake actions 