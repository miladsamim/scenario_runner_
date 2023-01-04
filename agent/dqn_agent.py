import numpy as np

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import os 
from timeit import default_timer as timer 

from . import MemoryBufferSimple, MemoryBufferSeparated

EVAL_FREQ=25
SAVE_FREQ=100

class DQN_Agent:
    # steer, throttle, brake | 0 SHIFTED FIRST
    action_space  =          [  [ 0.,     0.,     0.,   ], [-0.825,  0.,     0.   ], [-0.75,   0.,     0.   ], [-0.625,  0.,     0.   ], 
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
    space_2_idx = {tuple(act):idx for idx, act in 
                                enumerate(action_space)}
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
                 num_frames, model_name, store_path, learning_rate=0.00025, discount=0.98, delta=1):
        self.steering_space = [-1,-0.825,-0.75,-0.625,-0.5,-0.375,-0.25, -0.125, 0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.825, 1]
        self.throttle_space = [0, 0.5, 1]
        self.brake_space = [0, 0.5, 1]
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
        self.model_path = store_path  
        with open(os.path.join(self.model_path, self.model_name + '_args.txt'), 'w') as f: 
            for key, value in architecture_args.items(): 
                f.write('%s:%s\n' % (key, value))

        # Training parameters setup
        self.discount = discount
        self.num_frames = num_frames
        self.replay_memory = MemoryBufferSeparated(num_frames=num_frames, max_buffer_sz=memory_capacity)
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

    def save_model(self, model_id=None, special_path=None):
        path = os.path.join(self.model_path, self.model_name) if not special_path else special_path
        if model_id and not special_path:
            path += f'_{model_id}.pt'
        else:
            path += '.pt'
        torch.save(self.dqn.state_dict(), path)
    
    def real_to_space_idx(self, space, val, delta):
       idx = len(space)-1
       for i in range(len(space)):
              if val - delta < space[i]:
                    idx = i 
                    break
       return idx

    def act_to_discrete_idx(self, act, steer_delta=0.125/2, delta=0.5/2):
        throttle_used = True if act['throttle'] > 0 else False 
            
        steering_idx = self.real_to_space_idx(self.steering_space, act['steer'], steer_delta)
        steer_val = self.steering_space[steering_idx]
        if throttle_used:
            throttle_idx = self.real_to_space_idx(self.throttle_space, act['throttle'], delta)
            throttle_val = self.throttle_space[throttle_idx]
            brake_val = 0
        else: 
            brake_idx = self.real_to_space_idx(self.brake_space, act['brake'], delta)
            brake_val = self.brake_space[brake_idx]
            throttle_val = 0
        sep_space_idxs = (steer_val, throttle_val, brake_val)
        return self.space_2_idx[sep_space_idxs]

    # Description: Performs one step of batch gradient descent on the DDQN loss function. 
    # Parameters:
    # - alpha: Number, the learning rate 
    # Output: None
    def experience_replay(self):
        states, actions, rewards, dones = next(iter(self.replay_memory_sampler))
        # flip seq to be at axis 0 and batch samples at axis 1
        # print(actions, rewards, dones)
        for i, state_sensor in enumerate(states): 
            states[i] = state_sensor.transpose_(0,1).to(self.device) # in place transpose
        actions = actions.unsqueeze(-1).to(self.device)
        rewards = rewards.unsqueeze(1).to(self.device)
        dones = dones.unsqueeze(1).to(self.device)
        # x_act = states[-1]
        # print(actions, actions.shape)
        # print(x_act, x_act.shape)
        # print((actions == x_act[-1].unsqueeze(-1)).all())
        # for i in range(17):
        #     if (actions == x_act[i].unsque    eze(-1)).all():
        #         print(i, "MATCHES")

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
        # for param in self.dqn.parameters():
        #     param.grad.data.clamp_(-1, 1)
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
                print(q_vals.cpu(), act_idx)
            action = self.act_idx_to_dict(act_idx)   
            print(action)
            return act_idx, action  

    def act_idx_to_dict(self, act_idx):
        act_l = self.action_space[act_idx]
        return {'steer': act_l[0], 'throttle': act_l[1], 'brake': act_l[2]}

    def sample_action(self, acc_rate=0.4):        
        use_throttle = random.random() < acc_rate#acc_rate
        if use_throttle:
            act_idx = random.choice([52,56]) # 52, 56 -> throttle 0.5, 1.0 
        else:
            act_idx = random.randint(0, len(self.action_space)-1)
        action = self.act_idx_to_dict(act_idx) 
        return act_idx, action 
    
    def add_experience(self, *args, **kwargs):
        self.replay_memory.add_experience(*args, **kwargs)    

    # Description: Updates the weights of the target network
    # Parameters: 	None
    # Output: 		None
    def update_fixed_target_weights(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

