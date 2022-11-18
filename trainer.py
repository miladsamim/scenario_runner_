import traceback
from mpi4py import MPI

from srunner.tools import dotdict
from scenario_runner import ScenarioRunner

import sys
import time
import pickle

scenario_args = dotdict()
scenario_args.host = '127.0.0.1'
scenario_args.port = '2000'
scenario_args.timeout = '10.0'
scenario_args.trafficManagerPort = '8000'
scenario_args.trafficManagerSeed = '0'
scenario_args.sync = True # is required when training
scenario_args.list = False
scenario_args.scenario = 'StraightDriving_3'
scenario_args.openscenario = None 
scenario_args.openscenarioparams = None 
scenario_args.route = None
# scenario_args.agent = '.\\srunner\\autoagents\\human_agent.py' # agent module location
# scenario_args.agent = '.\\srunner\\autoagents\\npc_agent.py' # agent module location
scenario_args.agent = '.\\srunner\\autoagents\\simple_agent.py' # agent module location
scenario_args.agentConfig = '.\\srunner\\autoagents\\simple_agent_config.txt' 
scenario_args.output = False
scenario_args.file = False 
scenario_args.junit = False 
scenario_args.json = False 
scenario_args.outputDir = '' 
scenario_args.configFile = ''
scenario_args.additionalScenario = '' 
scenario_args.debug = True 
scenario_args.reloadWorld = True 
scenario_args.record = '' 
scenario_args.randomize = False 
scenario_args.repetitions = 1
scenario_args.waitForEgo = False
scenario_args.is_learner = True  

agent_config = dotdict({
            'sensor_setup': 'hd_map',
            'image_width': '96',
            'image_height': '96',
            'visualize_sensors': '1',
            'external_visualizer': '1',
            'fill_buffer': '0',
            'file': 'test.json',
        })

# SCENARIO_SPAWNER = 'mpi_tests/child.py'#scenario_spawner.py'
SCENARIO_SPAWNER = 'scenario_spawner.py'

class CarlaEnv:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    name = MPI.Get_processor_name()
    status = MPI.Status()
    def __init__(self, scenario_specification, agentConfig):
        self.scenario_specification = scenario_specification
        self.icomm = None 
        self.agentConfig = agentConfig
        if agentConfig:
            self._setup_agent()
    
    def _run_scenario(self):
        scenario_runner = None
        result = True
        self._setup_scenario()
        try:
            print("Building scenario")
            self.icomm = MPI.COMM_SELF.Spawn(sys.executable, args=[SCENARIO_SPAWNER], maxprocs=1, root=0)
            self.icomm.send(self.scenario_specification.copy(), dest=0, tag=11)
        except: 
            ValueError('Failed to build scenario / Spawn failed to start')
    
    def reset_env(self):
        """Reset sends message to destroy current scenario and starts a new one"""
        if self.icomm:
            data = {'reset': True,
                    'action': None} 
            print("Send prev message")
            self.icomm.send(data, dest=0, tag=1)
        self._run_scenario()
    
    def step(self, action, additional=None):
        """Action is a dict array specifying a subset of:
           'throttle':  [0.0, 1.0]  (defalt: 0)
           'steer':     [-1.0, 1.0] (defalt: 0)
           'brake'      [0.0, 1.0]  (defalt: 0)
           'hand_brake' bool        (default: False)
           'reverse'    bool        (default: False)
           'manual'     bool        (defalt: False)
           'gear'       int         (defalt: 0)"""

        # take action and receive s',r,d
        data = {'action': action,
                'reset':False} 
        data = self.icomm.sendrecv(data, dest=0, sendtag=2, source=0, recvtag=MPI.ANY_TAG)
        sensor_data = data['sensor_data']
        criterias = data['criterias']
        done = data['done']
        velocity = data['velocity']
        reward = self._compute_reward(criterias)
        state = self._process_state(sensor_data, velocity)
        with open('trainer_receive.txt', mode='w') as fp:
            fp.writelines(time.asctime()+'\n')
            fp.writelines(f'Done: {str(done)}\n')
            fp.writelines(f'Velocity: {velocity}\n')
        print(criterias)
        # print(sensor_data.keys())
        print(sensor_data)
            # save dictionary to pickle file
        with open('state.pickle', 'wb') as fp:
            pickle.dump(state, fp, protocol=pickle.HIGHEST_PROTOCOL)

        return state, reward, done

    def _process_state(self, sensor_data, velocity):
        assert self.agentConfig.sensor_setup in ['hd_map', 'none']
        processed_data = {}

        flip = lambda x: x[:,:,2::-1]

        processed_data['frame'] = sensor_data['IMU'][0]
        processed_data['accelerometer'] = sensor_data['IMU'][1][:3]
        processed_data['gyroscope'] = sensor_data['IMU'][1][3:6]
        processed_data['compass'] = sensor_data['IMU'][1][6:7]
        processed_data['gnss'] = sensor_data['GPS'][1]
        processed_data['velocity'] = velocity

        if self.agentConfig.sensor_setup == 'hd_map':
            processed_data['hd_map'] = flip(sensor_data['bev_sem'][1])/255.0
            processed_data['front_rgb'] = flip(sensor_data['front_rgb'][1])/255.0
        
        return processed_data

    def _compute_reward(self, criterias):
        return -0.1  

    def _setup_scenario(self):
        prev_path = self.scenario_specification.agentConfig
        self.scenario_specification.useMPI = True
        self.scenario_specification.agentConfig = self._agent_config_path if self.agentConfig else prev_path

    def _setup_agent(self):
        model_path = '.\\model_store\\' + self.agentConfig['sensor_setup'] + '_' +\
                     self.agentConfig['image_width'] + 'w_' + self.agentConfig['image_height'] + 'h_'
        self.agentConfig['model_path'] = model_path
        new_path = self.scenario_specification.agentConfig[:-4] + '_new.txt'
        with open(new_path, 'w') as fp:
            for key, value in self.agentConfig.items(): 
                fp.write('%s: %s\n' % (key, value))
        self._agent_config_path = new_path

class Trainer:
    def __init__(self, scenario_specifications, agentConfig, episodes, debug=False):
        self.scenario_specifications = scenario_specifications
        self.episodes = episodes 
        self.scenario_specifications.debug = debug 
        self.agentConfig = agentConfig 

        self.env = CarlaEnv(scenario_specifications, agentConfig)
        
    def _prefill_buffer(self):
        pass

    def train(self):
        self.env.reset_env()
        _,_,done = self.env.step(None)
        while not done:
            action = {'throttle':.6}
            _,_,done = self.env.step(action)

    def _monitor_criterias(self):
        pass 
    
if __name__ == '__main__':
    print("Starting training of agent")
    trainer = Trainer(scenario_args, agent_config, 1, debug=False)
    trainer.train()
    scenario_args = trainer.scenario_specifications
    # sys.exit(trainer.train())
    # try:
    #     scenario_runner = ScenarioRunner(scenario_args)
    #     result = scenario_runner.run()
    # except Exception:   # pylint: disable=broad-except
    #     traceback.print_exc()
    # finally:
    #     if scenario_runner is not None:
    #         print("destroyed")
    #         scenario_runner.destroy()
    #         del scenario_runner