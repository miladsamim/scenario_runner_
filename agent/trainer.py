from env import CarlaEnv
from agent import DQN_Agent, Decay_Explore_Rate
from model import HDDriveDQN, hd_net_args
from srunner.tools import dotdict
import random
from torch.utils.tensorboard import SummaryWriter
import os

# import traceback
from collections import deque

class ScenarioSampler:
    def __init__(self, default_spec):
        self.default_spec = default_spec
        self.scenarios = ['StraightDriving_1', 'StraightDriving_2', 'StraightDriving_4']
        self.npc_rate = 0.3 
    
    def sample(self, test=False):
        scenario_spec = dotdict(self.default_spec.copy())
        scenario_spec.scenario = random.choice(self.scenarios)
        if test:
            use_npc = False 
        else:
            use_npc = True if random.random() < self.npc_rate else False
        return scenario_spec, use_npc

class Trainer:
    def __init__(self, scenario_specification, agentConfig, 
                 model, model_args, exp_args, explore_rate, debug=False):
        self.scenario_specification = scenario_specification
        self.scenario_sampler = ScenarioSampler(scenario_specification)
        self.scenario_specification.debug = debug 
        self.agentConfig = agentConfig
        self.agent = DQN_Agent(model, model_args, exp_args.BATCH_SZ,
                               exp_args.REPLARY_MEM_SZ, exp_args.NUM_FRAMES,
                               exp_args.MODEL_NAME)
        self.exp_args = exp_args        
        self.expore_rate = explore_rate()
        if exp_args.LOAD_MODEL:
            print("Loaded existing model")
            self.agent.load_model(self.agent.model_path + exp_args.MODEL_NAME + '.pt')
        self.metrics_path = os.path.join('agent', 'model_store')
        self.writer = SummaryWriter(os.path.join(self.metrics_path, 'log', 'TEST/' if exp_args.TEST else 'TRAIN/')) 

        self.env = self._build_env()

    def _prefill_buffer(self):
        pass

    def _build_env(self, reset=False):
        if reset:
            del self.env 
        return CarlaEnv(self.scenario_specification, self.agentConfig)

    def _check_early_termination(self, info):
        """Will check for early termination cases:
           - actor isBlocked > 10 sec
           - actor offRoadTime > 10 sec"""
        early_termination = False 
        early_termination = early_termination or info['isBlocked']
        early_termination = early_termination or info['offRoadTime'] > 10
        return early_termination

    def train(self):
        exp_args = self.exp_args
        ep_i = 0
        step = 0
        actual_step = exp_args.START_STEP
        while actual_step < exp_args.NUM_STEPS:
            state_frame_stack = deque(maxlen=exp_args.NUM_FRAMES)
            ep_i += 1
            ep_reward = 0
            ep_steps = 0
            ep_collisions = 0
            ep_wronglanes = 0
            ep_offroad = 0 
            ep_rc_meters = 0

            scenario_spec, use_npc = self.scenario_sampler.sample(test=exp_args.TEST)
            self.env._set_scenario_spec(scenario_spec)
            self.env.reset_env()
            
            state, reward, done, info, terminated = self.env.step(None, early_termination=False, use_npc=use_npc)
            early_terminate = False 
            for i in range(exp_args.NUM_FRAMES):
                state_frame_stack.append(state)
            
            while not done:
                step += 1
                actual_step += 1
                # action = {'throttle':1.0}
                if use_npc:
                    action = None 
                else:
                    eps = self.expore_rate.get(step, exp_args.NUM_STEPS) if not exp_args.TEST else 0
                    action_idx, action = self.agent.get_split_action(state_frame_stack, epsilon=eps)
                
                state, reward, done, info, terminated = self.env.step(action, early_terminate, use_npc=use_npc)
                early_terminate = self._check_early_termination(info)
                if not terminated: # the scenario have crashed or finished
                    if use_npc: # convert npc action proper agent action 
                        action_idx = self.agent.act_to_discrete_split(info['npc_act'])
                    else: # don't log npc data
                        ep_reward += reward 
                        ep_steps += step
                        ep_collisions += info['isCollided']
                        ep_wronglanes = info['wrongLaneCount']
                        ep_offroad = info['outsideDrivingLanesCount']
                        ep_rc_meters = info['routeCompletionMeters']
                        self.writer.add_scalar('Reward/step', reward, step)
                        self.writer.add_scalar('Epsilon/step', eps, step)      
                    self.agent.add_experience(state, action_idx, reward, done)
                 
                    if not exp_args.TEST and self.agent.replay_memory.__len__() > exp_args.BATCH_SZ: # train when buffer is larger than batch size
                        self.agent.experience_replay_split_acts()


            # log episode metrics 
            self.writer.add_scalar('Steps/ep', ep_steps, ep_i)
            if not use_npc:
                self.writer.add_scalar('Reward/ep', ep_reward, ep_i)
                self.writer.add_scalar('ep_collisions', ep_collisions, ep_i)
                self.writer.add_scalar('ep_wronglanes', ep_wronglanes, ep_i)
                self.writer.add_scalar('ep_offroad', ep_offroad, ep_i)
                self.writer.add_scalar('ep_rc_meters', ep_rc_meters, ep_i)
            
            if not exp_args.TEST:
                self.agent.save_model()

    def _monitor_criterias(self):
        pass 
    
