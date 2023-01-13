from env import CarlaEnv
from agent import DQN_Agent
from srunner.tools import dotdict
import random
from timeit import default_timer as timer 
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os

# import traceback
from collections import deque

# TEST SCENARIOS
t_straight = ['PTESTStraight_1', 'PTESTStraight_2']
t_left_curved = ['TESTLeftCurved_1', 'TESTLeftCurved_2']
t_right_curved = ['TESTRightCurved_1', 'TESTRightCurved_2']
t_junction_3_left = ['PTEST3Left_1', 'PTEST3Left_2']
t_junction_3_right = ['PTEST3Right_1', 'PTEST3Right_2']
t_junction_4_left = ['PTEST4Left_1', 'PTEST4Left_2']
t_junction_4_straight = ['PTEST4Straight_1', 'PTEST4Straight_2']
t_junction_4_right = ['PTEST4Right_1', 'PTEST4Right_2']
test_straight_scenarios = {'straight': t_straight}
test_curved_scenarios = {'left_curved': t_left_curved,
                         'right_curved': t_right_curved}
test_junction_3 = {'junction_3_left': t_junction_3_left,
                   'junction_3_right': t_junction_3_right}
test_junction_4 = {'junction_4_left': t_junction_4_left,
                   'junction_4_straight': t_junction_4_straight,
                   'junction_4_right': t_junction_4_right}
test_scenarios = {'test_straight': test_straight_scenarios,
                  'test_curved': test_curved_scenarios,
                  'test_junction_3': test_junction_3,
                  'test_junction_4': test_junction_4}

# TRAIN SCENARIOS
# STRAIGHT
straight = ['StraightDriving_1', 'StraightDriving_2', 'StraightDriving_4']
# CURVED
right_curved = ['RightCurved_1', 'RightCurved_2', 'RightCurved_3']
left_curved = ['LeftCurved_1', 'LeftCurved_2', 'LeftCurved_3', 'PLeftCurved_1']
# 3-HEADED JUNCTIONS
junction_3_left = ['P3Left_1', 'P3Left_2', 'P3Left_3', 'P3Left_4']
junction_3_straight = ['P3Straight_1']
junction_3_right = ['P3Right_1', 'P3Right_2', 'P3Right_3', 'P3Right_4', 'P3Right_5']
# 4-HEADED JUNCTIONS 
junction_4_left = ['P4Left_1', 'P4Left_2'] # 'P4LR_1'
junction_4_straight = ['P4Straight_1', 'P4Straight_2', 'P4Straight_3']
junction_4_right = ['P4Right_1', 'P4Right_2', 'P4LR_2']
train_straight_scenarios = {'straight': straight}
train_curved_scenarios = {'left_curved': left_curved,
                         'right_curved': right_curved}
train_junction_3 = {'junction_3_left': junction_3_left,
                   'junction_3_straight': junction_3_straight,
                   'junction_3_right': junction_3_right}
train_junction_4 = {'junction_4_left': junction_4_left,
                    'junction_4_straight': junction_4_straight,
                    'junction_4_right': junction_4_right}
train_scenarios = {'straight': train_straight_scenarios,
                   'curved': train_curved_scenarios,
                   'junction_3': train_junction_3,
                   'junction_4': train_junction_4}

class ScenarioSampler:
    straight = straight
    curved_scenarios = left_curved + right_curved
    
    junction_3 = junction_3_left + junction_3_straight + junction_3_right
    junction_4 = junction_4_left + junction_4_straight + junction_3_right

    junctions = junction_3 + junction_4

    # LEGS
    leg2_scenarios = straight + curved_scenarios
    leg3_scenarios = leg2_scenarios + junctions 

    def __init__(self, default_spec, test_regiment=False):
        self.default_spec = default_spec
        self.scenarios = ['PTEST4Right_1']#self.leg3_scenarios
        self.npc_rate = 0.25
        self.test_regiment = True 

    def build_spec(self, scenario_name):
        scenario_spec = dotdict(self.default_spec.copy())
        scenario_spec.scenario = scenario_name 
        return scenario_spec


    def sample(self, test=False):
        scenario_spec = dotdict(self.default_spec.copy())
        scenario_spec.scenario = random.choice(self.scenarios)
        if test:
            use_npc = False 
        else:
            use_npc = True if random.random() < self.npc_rate else False
        acc_rate = random.random() # rate of frequency when throttle-only actions are chosen
        return scenario_spec, use_npc, acc_rate

class Trainer:
    def __init__(self, scenario_specification, agentConfig, 
                 model, model_args, exp_args, explore_rate, debug=False):
        self.scenario_specification = scenario_specification
        self.scenario_sampler = ScenarioSampler(scenario_specification)
        self.scenario_specification.debug = debug 
        self.agentConfig = agentConfig

        self.model_name = model.__name__ + f'_{model_args.n_frames}f_' + f'{model_args.residual}res' + '_leg4'
        # self.model_name = model.__name__ + f'_{model_args.n_frames}f_' + f'{model_args.residual}res' + '_LEG4_ALL'

        if exp_args.EVAL:
            self.model_store_path = os.path.join('agent', 'model_store', 'ORG')
        else:
            self.model_store_path = os.path.join('agent', 'model_store', self.model_name)
        self.writer = SummaryWriter(os.path.join(self.model_store_path, 'log', 'TEST/' if exp_args.TEST else 'TRAIN/')) 
        self.start_time = timer()
        self.agent = DQN_Agent(model, model_args, exp_args.BATCH_SZ,
                               exp_args.REPLARY_MEM_SZ, exp_args.NUM_FRAMES,
                               self.model_name, self.model_store_path)
        self.exp_args = exp_args        
        self.expore_rate = explore_rate()
        if exp_args.LOAD_MODEL:
            print("Loaded existing model")
            self.agent.load_model(os.path.join(self.agent.model_path,  self.model_name + '_128.pt'))
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

    def evaluate(self):
        exp_args = self.exp_args
        stats = {'category':[],
                 'type': [],
                 'name': [],
                 'collisionCount': [],
                 'wrongLaneCount': [],
                 'outsideDrivingLanesCount': [],
                 'offRoadCount': [],
                 'routeCompletionPer': []}

        scenarios_dict = {**test_scenarios, **train_scenarios} 

        for variant_category, variant_scenarios in scenarios_dict.items():
            for variant_type, variants in variant_scenarios.items():
                for variant in variants:
                    for i in range(exp_args.EVAL_ROUNDS): # its
                        ep_reward = 0 
                        scenario_spec = self.scenario_sampler.build_spec(variant)
                        use_npc, acc_rate, eps = False, 0, 0
                        self.env._set_scenario_spec(scenario_spec)
                        self.env.reset_env()

                        state, reward, done, info, terminated = self.env.step(None, early_termination=False, use_npc=use_npc)
                        if terminated: # move to next ep on crash
                            continue
                        state['action_idx'] = 0 
                        early_terminate = False 
                        state_frame_stack = deque(maxlen=exp_args.NUM_FRAMES)
                        for _ in range(exp_args.NUM_FRAMES):
                            state_frame_stack.append(state)

                        while not done:
                            action_idx, action = self.agent.get_action(state_frame_stack, epsilon=eps, acc_rate=acc_rate)
                            new_state, reward, done, info, terminated = self.env.step(action, early_terminate, use_npc=use_npc)
                            early_terminate = self._check_early_termination(info)
                            if not terminated: # the scenario have crashed or finished
                                ep_reward += reward
                                new_state['action_idx'] = action_idx
                                state = new_state
                                state_frame_stack.append(state)

                        stats['category'].append(variant_category)
                        stats['type'].append(variant_type)
                        stats['name'].append(variant)
                        stats['collisionCount'].append(info['collisionCount'])
                        stats['wrongLaneCount'].append(info['wrongLaneCount'])
                        stats['outsideDrivingLanesCount'].append(info['outsideDrivingLanesCount'])
                        stats['offRoadCount'].append(info['offRoadCount'])
                        stats['routeCompletionPer'].append(info['routeCompletionPer'])

        # write down stats
        df_stats = pd.DataFrame(stats)
        df_stats.index = pd.Index(range(len(df_stats)), name='idx')
        file_path = os.path.join(self.model_store_path, f'eval_stats_{self.model_name}.csv')
        df_stats.to_csv(file_path)



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

            test = True if ep_i % exp_args.TEST_EVERY == 0 else exp_args.TEST

            scenario_spec, use_npc, acc_rate = self.scenario_sampler.sample(test=test)
            self.env._set_scenario_spec(scenario_spec)
            self.env.reset_env()
            
            state, reward, done, info, terminated = self.env.step(None, early_termination=False, use_npc=use_npc)
            if terminated: # move to next ep on crash
                continue
            state['action_idx'] = 0 
            early_terminate = False 
            for i in range(exp_args.NUM_FRAMES):
                state_frame_stack.append(state)
            self.agent.add_experience(state, 0, reward, done, new_episode=True)
            
            while not done:
                step += 1
                actual_step += 1
                if use_npc:
                    action = None 
                    reward = 1 # add extra bonus 
                else:
                    eps = self.expore_rate.get(actual_step, exp_args.EXPLORE_STEPS) if not test else 0
                    action_idx, action = self.agent.get_action(state_frame_stack, epsilon=eps, acc_rate=acc_rate)

                new_state, reward, done, info, terminated = self.env.step(action, early_terminate, use_npc=use_npc)
                early_terminate = self._check_early_termination(info)
                if not terminated: # the scenario have crashed or finished
                    if use_npc: # convert npc action proper agent action 
                        action_idx = self.agent.act_to_discrete_idx(info['npc_act'])
                        # print(info['npc_act'], self.agent.action_space[action_idx])
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
                    new_state['action_idx'] = action_idx
                    state = new_state
                    state_frame_stack.append(state)
                    if step % exp_args.TARGET_UPDATE_FREQ == 0:
                        self.agent.update_fixed_target_weights()
                    if not test and self.agent.replay_memory.__len__() > exp_args.BATCH_SZ: # train when buffer is larger than batch size
                        self.agent.experience_replay()

            # log episode metrics 
            self.writer.add_scalar('Steps/ep', ep_steps, ep_i)
            episode_end_time = timer() - self.start_time 
            self.writer.add_scalar('Time', episode_end_time, ep_i)
            if not use_npc:
                b_text = 'TEST/' if test else '' 
                self.writer.add_scalar(b_text+'Reward/ep', ep_reward, ep_i)
                self.writer.add_scalar(b_text+'ep_collisions', ep_collisions, ep_i)
                self.writer.add_scalar(b_text+'ep_wronglanes', ep_wronglanes, ep_i)
                self.writer.add_scalar(b_text+'ep_offroad', ep_offroad, ep_i)
                self.writer.add_scalar(b_text+'ep_rc_meters', ep_rc_meters, ep_i)
            
            if not exp_args.TEST:
                self.agent.save_model(model_id=ep_i//exp_args.STORE_DENSITY)

    def _monitor_criterias(self):
        pass 
    
