import numpy as np

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter

import random 
import os 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_FREQ=25
SAVE_FREQ=100

class DQN_Agent:

	# Description: Initializes the DQN_Agent object
	# Parameters:
	# - environment: 		Object supporting methods like 'reset', 'render', 'step' etc.
	# 			     		For an example see environment.py.
	# - architecture: 		Object supporting the method 'evaluate', usually a neural network. 
	# 				  		for exapmles see parameters/architectures.py.
	# - explore_rate: 		Object supporting the method 'get'. See parameters/explore_rates.py for examples.
	# - learning_rate: 		Object supporting the method 'get'. See parameters/learning_rates.py for examples.
	# - batch_size: 		Integer giving the size of the minibatch to be used in the optimization process.
	# - memory_capacity: 	Integer giving the size of the replay memory.
	# - num_episodes: 		Integer specifying the number of training episodes to be used. 
	# - learning_rate_drop_frame_limit:
	# 						Integer specifying by which frame during training the minimal explore rate should be reached 
	# - target_update_frequency:
	# 						Integer specifying the frequency at which the target network's weights are to be updated
	# - discount:			Number in (0,1) giving the discount factor
	# - delta:  			Number, the delta parameter in huber loss
	# - model_name:  		String, the name of the folder where the model is saved. 
	# Output: None

    # Tasks 
    # experience_replay
    # hold the buffer 
    # get_action
    # load model
    # save model

    def __init__(self, environment, architecture, architecture_args, explore_rate, learning_rate,
                 batch_size, memory_capacity, num_episodes, learning_rate_drop_frame_limit,
                 target_update_frequency, discount=0.99, delta=1, model_name=None):
        self.action_size = self.env.action_space_size
        self.avg_reward = None
        self.dqn = architecture(*architecture_args).to(DEVICE)
        self.target_dqn = architecture(*architecture_args).to(DEVICE)
        self.update_fixed_target_weights()
        self.learning_rate = 0.00025# learning_rate() # atari learning rate
        self.optim = torch.optim.Adam(self.dqn.parameters(), lr=self.learning_rate)
        self.explore_rate = explore_rate()
        self.criterion = nn.HuberLoss()

        self.model_path = os.path.dirname(os.path.realpath(__file__)) + '/models/' + model_name if model_name else str(self.env)
        self.log_path = self.model_path + '/log'
        self.writer = SummaryWriter(self.log_path)

        # Training parameters setup
        self.target_update_frequency = target_update_frequency
        self.discount = discount
        self.replay_memory = rplm.Replay_Memory(memory_capacity, batch_size)
        # self.training_metadata = utils.Training_Metadata(frame=self.sess.run(self.frames), frame_limit=learning_rate_drop_frame_limit,
        # 												   episode=self.sess.run(self.episode), num_episodes=num_episodes)
        self.training_metadata = utils.Training_Metadata(frame=0, frame_limit=learning_rate_drop_frame_limit,
                                                         episode=0, num_episodes=num_episodes)
        self.delta = delta
        utils.document_parameters(self)
        self.start_time = timer()

    # Description: Updates the weights of the target network
    # Parameters: 	None
    # Output: 		None
    def update_fixed_target_weights(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

