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
hd_net_args.n_frames = 12
hd_net_args.act_steering_dim = 11 # number of discrete steering actions
hd_net_args.act_throttle_dim = 11 # number of discrete throttle actions 
hd_net_args.act_brake_dim = 11 # number of discrete brake actions
hd_net_args.simple_space = True # whether to use standard discrete action space 
hd_net_args.simple_space_dim = 85 # number of discrete actions in action_space
hd_net_args.residual = True # whether to add a residual connection from state processing to post temporal net

# steer, throttle, brake | 0 SHIFTED FIRST
hd_net_args.action_space  =  [  [ 0.,     0.,     0.,   ], [-0.825,  0.,     0.   ], [-0.75,   0.,     0.   ], [-0.625,  0.,     0.   ], 
                                [-1.,     0.,     0.5,  ], [-0.825,  0.,     0.5  ], [-0.75,   0.,     0.5  ], [-0.625,  0.,     0.5  ],
                                [-1.,     0.,     1.,   ], [-0.825,  0.,     1.   ], [-0.75,   0.,     1.   ], [-0.625,  0.,     1.   ],
                                [-1.,     0.5,    0.,   ], [-0.825,  0.5,    0.   ], [-0.75,   0.5,    0.   ], [-0.625,  0.5,    0.   ],
                                [-1.,     1.,     0.,   ], [-0.825,  1.,     0.   ], [-0.75,   1.,     0.   ], [-0.625,  1.,     0.   ],
                                
                                [-0.5,    0.,     0.   ], [-0.375,  0.,     0.   ], [-0.25,   0.,     0.   ], [-0.125,  0.,     0.   ], 
                                [-0.5,    0.,     0.5  ], [-0.375,  0.,     0.5  ], [-0.25,   0.,     0.5  ], [-0.125,  0.,     0.5  ],
                                [-0.5,    0.,     1.   ], [-0.375,  0.,     1.   ], [-0.25,   0.,     1.   ], [-0.125,  0.,     1.   ],
                                [-0.5,    0.5,    0.   ], [-0.375,  0.5,    0.   ], [-0.25,   0.5,    0.   ], [-0.125,  0.5,    0.   ],
                                [-0.5,    1.,     0.   ], [-0.375,  1.,     0.   ], [-0.25,   1.,     0.   ], [-0.125,  1.,     0.   ],
                                
                                [-1.,     0.,     0.   ], [ 0.125,  0.,     0.   ], [ 0.25,   0.,     0.   ], [ 0.375,  0.,     0.   ],  
                                [ 0.,     0.,     0.5  ], [ 0.125,  0.,     0.5  ], [ 0.25,   0.,     0.5  ], [ 0.375,  0.,     0.5  ],  
                                [ 0.,     0.,     1.   ], [ 0.125,  0.,     1.   ], [ 0.25,   0.,     1.   ], [ 0.375,  0.,     1.   ],  
                                [ 0,     0.5,     0.   ], [ 0.125,  0.5,    0.   ], [ 0.25,   0.5,    0.   ], [ 0.375,  0.5,    0.   ],  
                                [ 0.,     1.,     0.   ], [ 0.125,  1.,     0.   ], [ 0.25,   1.,     0.   ], [ 0.375,  1.,     0.   ],  

                                [ 0.5,    0.,     0.   ], [ 0.625,  0.,     0.   ], [ 0.75,   0.,     0.   ], [ 0.825,  0.,     0.   ], 
                                [ 0.5,    0.,     0.5  ], [ 0.625,  0.,     0.5  ], [ 0.75,   0.,     0.5  ], [ 0.825,  0.,     0.5  ],
                                [ 0.5,    0.,     1.   ], [ 0.625,  0.,     1.   ], [ 0.75,   0.,     1.   ], [ 0.825,  0.,     1.   ],
                                [ 0.5,    0.5,    0.   ], [ 0.625,  0.5,    0.   ], [ 0.75,   0.5,    0.   ], [ 0.825,  0.5,    0.   ],
                                [ 0.5,    1.,     0.   ], [ 0.625,  1.,     0.   ], [ 0.75,   1.,     0.   ], [ 0.825,  1.,     0.   ],  

                                [ 1.,     0.,     0.   ],
                                [ 1.,     0.,     0.5  ],
                                [ 1.,     0.,     1.   ],
                                [ 1.,     0.5,    0.   ],
                                [ 1.,     1.,     0.   ]]
#| 0 SHIFTED FIRST
hd_net_args.space_2_idx = {tuple(act):idx for idx, act in 
                            enumerate(hd_net_args.action_space)}