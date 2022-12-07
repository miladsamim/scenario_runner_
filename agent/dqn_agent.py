import numpy as np

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import os 
from timeit import default_timer as timer 

from . import MemoryBufferSimple

EVAL_FREQ=25
SAVE_FREQ=100

class DQN_Agent:

    # 35 discrete actions
    # steer in (-0.6, 0.3, 0.0, 0.3 0.6) | throttle/brake in (0.0, 0.3, 0.6, 0.9) 
    action_space = [[0.0,  0, 0.0],
                    [-0.6, 0.3, 0], [-0.6, 0.6, 0], [-0.6, 0.9, 0],
                    [-0.3, 0.3, 0], [-0.3, 0.6, 0], [-0.3, 0.9, 0],
                    [ 0.0, 0.3, 0], [ 0.0, 0.6, 0], [ 0.0, 0.9, 0],
                    [ 0.3, 0.3, 0], [ 0.3, 0.6, 0], [ 0.3, 0.9, 0],
                    [ 0.6, 0.3, 0], [ 0.6, 0.6, 0], [ 0.6, 0.9, 0],

                    [-0.6, 0, 0.3], [-0.3, 0, 0.3], [ 0.0, 0, 0.3], [ 0.3, 0, 0.3],
                    [ 0.6, 0, 0.3], [-0.6, 0, 0.6], [-0.3, 0, 0.6], [ 0.0, 0, 0.6],
                    [ 0.3, 0, 0.6], [ 0.6, 0, 0.6], [-0.6, 0, 0.9], [-0.3, 0, 0.9],
                    [ 0.0, 0, 0.9], [ 0.3, 0, 0.9], [ 0.6, 0, 0.9]]

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

    def __init__(self, architecture, architecture_args, batch_size, memory_capacity,
                 num_frames, model_name, learning_rate=0.00025, discount=0.99, delta=1):
        # self.action_size = self.env.action_space_size
        self.steering_act_space = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.throttle_act_space = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.brake_act_space = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.avg_reward = None
        self.device = architecture_args.device
        self.dqn = architecture(architecture_args).to(self.device)
        self.target_dqn = architecture(architecture_args).to(self.device)
        self.update_fixed_target_weights()
        self.learning_rate = learning_rate # 0.00025# learning_rate() # atari learning rate
        parameters = [ param for param in self.dqn.parameters() if param.requires_grad == True]
        self.optim = torch.optim.Adam(parameters, lr=self.learning_rate)
        # self.explore_rate = explore_rate()
        self.criterion = nn.HuberLoss()
        self.model_name = model_name
        self.model_path = os.path.dirname(os.path.realpath(__file__)) + '\\model_store\\' 
        with open(self.model_path + model_name + '_args.txt', 'w') as f: 
            for key, value in architecture_args.items(): 
                f.write('%s:%s\n' % (key, value))

        # Training parameters setup
        self.discount = discount
        self.num_frames = num_frames
        self.replay_memory = MemoryBufferSimple(num_frames=num_frames, max_buffer_sz=memory_capacity)
        self.replay_memory_sampler = torch.utils.data.DataLoader(self.replay_memory, batch_size=batch_size)
        # self.training_metadata = utils.Training_Metadata(frame=self.sess.run(self.frames), frame_limit=learning_rate_drop_frame_limit,
        # 												   episode=self.sess.run(self.episode), num_episodes=num_episodes)

        self.delta = delta
        self.start_time = timer()
    
    def get_model_name(self, args):
        model_name = ''
        for key, val in args.items():
            model_name += f'{key}_{val}_'
        return model_name + '.pt'
    
    def load_model(self, path):
        self.dqn.load_state_dict(torch.load(path))
        self.update_fixed_target_weights()

    def save_model(self, path=None):
        path = self.model_path + self.model_name + '.pt' if not path else path
        torch.save(self.dqn.state_dict(), path)
    
    def real_to_space_idx(self, space, val, delta):
       idx = len(space)-1
       for i in range(len(space)):
              if val - delta < space[i]:
                     idx = i 
                     break
       return idx

    def act_to_discrete_split(self, act, steer_delta=0.2/2, delta=0.1/2):
        throttle_used = True if act['throttle'] > 0 else False 
        steering_idx = self.real_to_space_idx(self.steering_act_space, act['steer'], steer_delta)
        if throttle_used:
            throttle_idx = self.real_to_space_idx(self.throttle_act_space, act['throttle'], delta)
            brake_idx = 0
        else: 
            brake_idx = self.real_to_space_idx(self.brake_act_space, act['brake'], delta)
            throttle_idx = 0
        return [steering_idx, throttle_idx, brake_idx]

    # Description: Performs one step of batch gradient descent on the DDQN loss function. 
    # Parameters:
    # - alpha: Number, the learning rate 
    # Output: None
    def experience_replay(self):
        states, actions, rewards, dones = next(iter(self.replay_memory_sampler))
        # flip seq to be at axis 0 and batch samples at axis 1
        for i, state_sensor in enumerate(states): 
            states[i] = state_sensor.transpose_(0,1).to(self.device) # in place transpose
        actions = actions.unsqueeze(-1).to(self.device)
        rewards = rewards.unsqueeze(1).to(self.device)
        dones = dones.unsqueeze(1).to(self.device)

        # the 0...n-1 frames marks the current state (t-num_frames,...,t-1,t)
        cur_states = [states_sensor[:-1] for states_sensor in states]
        # 1...n frames marks the next (t+1-num_frames,...,t-1,t,t+1)
        next_states = [states_sensor[1:] for states_sensor in states]

        with torch.no_grad():
            self.dqn.eval() # don't use dropout when estimating targets
            self.target_dqn.eval()
            greedy_actions = self.dqn(*next_states).argmax(dim=1, keepdims=True)
            q_value_targets = rewards + self.discount * ((1 - dones) * self.target_dqn(*next_states))
            q_value_targets = q_value_targets.gather(1, greedy_actions)
        
        self.dqn.train()
        q_value = self.dqn(*cur_states)
        q_value = q_value.gather(1, actions)

        loss = self.criterion(q_value, q_value_targets)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def experience_replay_split_acts(self):
        """States: is a tuple of each sensor measure, which itself is a tensor
           actions: is a tensor where each sample represents list of indicies which maps to the respective action spaces
           rewards: is a tensor of the float32 reward
           dones: is a tensor of boolean (1/0 in float32) which represents whether the episode terminated at that timestep"""
        states, actions, rewards, dones = next(iter(self.replay_memory_sampler))
        # flip seq to be at axis 0 and batch samples at axis 1
        for i, state_sensor in enumerate(states): 
            states[i] = state_sensor.transpose_(0,1).to(self.device) # in place transpose
        actions = actions.unsqueeze(-1).to(self.device)
        rewards = rewards.unsqueeze(1).to(self.device)
        dones = dones.unsqueeze(1).to(self.device)

        # the 0...n-1 frames marks the current state (t-num_frames,...,t-1,t)
        cur_states = [states_sensor[:-1] for states_sensor in states]
        # 1...n frames marks the next (t+1-num_frames,...,t-1,t,t+1)
        next_states = [states_sensor[1:] for states_sensor in states]
        
        with torch.no_grad():
            self.dqn.eval() # don't use dropout when estimating targets
            self.target_dqn.eval() 
            
            # Get greedy argmax actions using current dqn
            greedy_actions = self.dqn(*next_states)
            greedy_steering_actions = greedy_actions[0].argmax(dim=1, keepdim=True)
            greedy_throttle_actions = greedy_actions[1].argmax(dim=1, keepdim=True)
            greedy_brake_actions = greedy_actions[2].argmax(dim=1, keepdim=True)
            greedy_t_or_b = greedy_actions[3]

            # estimate q_vals using target dqn 
            steering_q_targets, throttle_q_targets, brake_q_targets, t_or_b_targets = self.target_dqn(*next_states)

            q_value_targets_steer = rewards + self.discount * ((1 - dones) * steering_q_targets)
            q_value_targets_steer = q_value_targets_steer.gather(1, greedy_steering_actions)

            q_value_targets_throttle = rewards + self.discount * ((1 - dones) * throttle_q_targets)
            q_value_targets_throttle = q_value_targets_throttle.gather(1, greedy_throttle_actions)

            q_value_targets_brake = rewards + self.discount * ((1 - dones) * brake_q_targets)
            q_value_targets_brake = q_value_targets_brake.gather(1, greedy_brake_actions)

            # compute q using linear combination of all action spaces
            q_value_targets = q_value_targets_steer + greedy_t_or_b * q_value_targets_throttle + (1-greedy_t_or_b)*q_value_targets_brake
        
        self.dqn.train()
        steering_q, throttle_q, brake_q, t_or_b = self.dqn(*cur_states)
        steering_q = steering_q.gather(1, actions[:,0])
        throttle_q = throttle_q.gather(1, actions[:,1])
        brake_q = brake_q.gather(1, actions[:,2])
        # compute q using linear combination of all action spaces
        q_value = steering_q + t_or_b * throttle_q + (1-t_or_b)*brake_q

        loss = self.criterion(q_value, q_value_targets)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        
    # Description: Chooses action wrt an e-greedy policy. 
    # Parameters:
    # - state: 		Tensor representing a single state
    # - epsilon: 	Number in (0,1)
    # Output: 		Integer in the range 0...self.action_size-1 representing an action
    def get_action(self, state, epsilon, acc_rate=0.4):
        """state: is a list of tuples of tensors which can be processed replay_memory._process_state"""
        # Performing epsilon-greedy action selection
        if random.random() < epsilon:
            return self.sample_action(acc_rate=acc_rate)
        else:
            state = self.replay_memory._process_states(state)
            temp = []
            for state_sensor in state:
                temp.append(state_sensor.unsqueeze(1).to(self.device))
            state = temp
            with torch.no_grad():
                self.dqn.eval()
                q_vals = self.dqn(*state)
                act_idx = q_vals.argmax(dim=1).item()
            action = self.act_idx_to_dict(act_idx)   
            print(q_vals.detach().cpu())
            return act_idx, action 

    def get_split_action(self, state, epsilon):
        """state: is a list of tuples of tensors which can be processed replay_memory._process_state"""
        # Performing epsilon-greedy action selection
        if random.random() < epsilon:
            return self.sample_split_action()
        else:
            state = self.replay_memory._process_states(state)
            temp = []
            for state_sensor in state:
                temp.append(state_sensor.unsqueeze(1).to(self.device))
            state = temp
            with torch.no_grad():
                self.dqn.eval()
                steering_q, throttle_q, brake_q, t_or_b = self.dqn(*state)
                use_throttle = t_or_b.item() > 0.5 
                steering_act = steering_q.argmax(dim=1).item() 
                throttle_act = throttle_q.argmax(dim=1).item() if use_throttle else 0
                brake_act = brake_q.argmax(dim=1).item() if not use_throttle else 0

                action_indicies = [steering_act, throttle_act, brake_act]
                action = {
                        'steer': self.steering_act_space[steering_act],
                        'throttle': self.throttle_act_space[throttle_act],
                        'brake': self.brake_act_space[brake_act],
                } 
                return action_indicies, action 

    def act_idx_to_dict(self, act_idx):
        act_l = self.action_space[act_idx]
        return {'steer': act_l[0], 'throttle': act_l[1], 'brake': act_l[2]}

    def sample_action(self, acc_rate=0.4):        
        use_throttle = random.random() < acc_rate
        if use_throttle:
            act_idx = random.choice([7,8,9]) # 7-9 -> throttle 0.3, 0.6, 0.9 
        else:
            act_idx = random.randint(0, len(self.action_space)-1)
        action = self.act_idx_to_dict(act_idx) 
        return act_idx, action 

    def sample_split_action(self, accelerated=False):        
        use_throttle = random.random() > 0.2 
        steering_act = random.choice(range(len(self.steering_act_space)))
        throttle_act = random.choice(range(len(self.throttle_act_space))) if use_throttle else 0
        brake_act = random.choice(range(len(self.brake_act_space))) if not use_throttle else 0
        
        if accelerated:
            throttle_act = random.choice(range(6, len(self.throttle_act_space))) # 0.6 -1 
            brake_act = 0 # 0

        action_indicies = [steering_act, throttle_act, brake_act]
        action = {
                'steer': self.steering_act_space[steering_act],
                'throttle': self.throttle_act_space[throttle_act],
                'brake': self.brake_act_space[brake_act],
        } 
        return action_indicies, action
    
    
    def add_experience(self, state:dict, action:list, reward:float, done:bool):
        self.replay_memory.add_experience(state, action, reward, done)    

    # Description: Updates the weights of the target network
    # Parameters: 	None
    # Output: 		None
    def update_fixed_target_weights(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

