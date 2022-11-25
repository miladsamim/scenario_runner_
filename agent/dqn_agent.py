import numpy as np

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter

import random 
import os 
from timeit import default_timer as timer 

from . import MemoryBufferSimple

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
                 batch_size, memory_capacity, num_frames, num_episodes, learning_rate_drop_frame_limit,
                 target_update_frequency, discount=0.99, delta=1, model_name=None):
        # self.action_size = self.env.action_space_size
        self.throttle_act_space = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.brake_act_space = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.steering_act_space = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
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
        self.num_frames = num_frames
        self.replay_memory = MemoryBufferSimple(num_frames=num_frames, max_buffer_sz=memory_capacity)
        self.replay_memory_sampler = torch.utils.data.DataLoader(self.replay_memory, batch_size=batch_size)
        self.memory_sampler = torch.utils.data.DataLoader()
        # self.training_metadata = utils.Training_Metadata(frame=self.sess.run(self.frames), frame_limit=learning_rate_drop_frame_limit,
        # 												   episode=self.sess.run(self.episode), num_episodes=num_episodes)

        self.delta = delta
        self.start_time = timer()

    
    # Description: Performs one step of batch gradient descent on the DDQN loss function. 
    # Parameters:
    # - alpha: Number, the learning rate 
    # Output: None
    def experience_replay(self):
        batch_X = next(iter(self.memory_sampler))
        
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_memory.get_mini_batch(self.training_metadata)
        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32).to(DEVICE)
        action_batch = torch.tensor(np.array(action_batch), dtype=torch.int64).argmax(dim=1,keepdim=True).to(DEVICE)
        reward_batch = torch.tensor(np.array(reward_batch), dtype=torch.float32).unsqueeze(1).to(DEVICE)
        next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32).to(DEVICE)
        done_batch = torch.tensor(np.array(done_batch), dtype=torch.float32).unsqueeze(1).to(DEVICE)

        with torch.no_grad():
            self.dqn.eval() # don't use dropout when estimating targets
            self.target_dqn.eval()
            greedy_actions = self.dqn(next_state_batch).argmax(dim=1, keepdims=True)
            q_value_targets = reward_batch + self.discount * ((1 - done_batch) * self.target_dqn(next_state_batch))
            q_value_targets = q_value_targets.gather(1, greedy_actions)
        
        self.dqn.train()
        q_value = self.dqn(state_batch)
        q_value = q_value.gather(1, action_batch)

        loss = self.criterion(q_value, q_value_targets)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    
    def add_experience(self, state:dict, action:list, reward:float, done:bool):
        self.replay_memory.add_experience(state, action, reward, done)    

    # Description: Updates the weights of the target network
    # Parameters: 	None
    # Output: 		None
    def update_fixed_target_weights(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

