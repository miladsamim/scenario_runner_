import torch 
import itertools
import numpy as np
from collections import deque 


class MemoryBufferSimple(torch.utils.data.Dataset):
    """Assumme that episodes are long, so probability that we will land in a part which
       starts a new episode is small, so just store in simple arrays"""
    def __init__(self, num_frames, max_buffer_sz=25_000):
        self.max_buffer_size = max_buffer_sz
        self.states = deque(maxlen=max_buffer_sz)
        self.actions = deque(maxlen=max_buffer_sz)
        self.rewards = deque(maxlen=max_buffer_sz)
        self.dones = deque(maxlen=max_buffer_sz)
        self.active_ep_idx = -1 
        self.num_frames = num_frames
    
    def __len__(self):
        return len(self.states) - self.num_frames - 1 # -1 as we need s_t+1 also

    def add_experience(self, state:dict, action:list, reward:float, done:bool):
        """states are at t+1, so when get_item we have to select actions, rewards one forward
           state>dict: which can be processed by _process_states
           action>int: which maps to action value in action_space list 
           reward>float: float value representing reward r(s'|s,a)
           done>bool: whether the episode is terminated at current step. Influences the target update"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def _np_img_to_tensor(self, img):
        img = np.array(img) # num_frames+1 x W x H x C
        return torch.tensor(img.transpose(2,0,1), dtype=torch.float32)

    def _process_states(self, states):
        """This function should be passed into the class as it could depend on the
           sensor setup. This configuration works for the HDSensor setup."""
        bev_x = []; front_x = []; acc_x = []; comp_x = []; gyr_x = []; vel_x = []; act_x = [];
        for state in states:
            bev_x.append(self._np_img_to_tensor(state['hd_map']))
            front_x.append(self._np_img_to_tensor(state['front_rgb']))
            acc_x.append(torch.tensor(state['accelerometer'], dtype=torch.float32))
            comp_x.append(torch.tensor(state['compass'], dtype=torch.float32))
            gyr_x.append(torch.tensor(state['gyroscope'], dtype=torch.float32))
            vel_x.append(torch.tensor(state['velocity'], dtype=torch.float32))
            act_x.append(torch.tensor(state['action_idx'], dtype=torch.long))

        bev_x = torch.stack(bev_x)
        front_x = torch.stack(front_x)
        acc_x = torch.stack(acc_x)
        comp_x = torch.stack(comp_x)
        gyr_x = torch.stack(gyr_x)
        vel_x = torch.stack(vel_x)
        act_x = torch.stack(act_x)

        return bev_x, front_x, acc_x, comp_x, gyr_x, vel_x, act_x
        
    def __getitem__(self, idx):
        end_idx = idx + self.num_frames + 1 # +1 as we need next state as well
        states = self._process_states(itertools.islice(self.states, idx, end_idx))
        action = torch.tensor(self.actions[end_idx], dtype=torch.int64)
        reward = torch.tensor(self.rewards[end_idx], dtype=torch.float32)
        dones = torch.tensor(self.dones[end_idx], dtype=torch.float32)
        return states, action, reward, dones