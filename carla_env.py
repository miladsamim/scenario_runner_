from mpi4py import MPI

import sys 
import time
import pickle 


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
        if self.scenario_specification.useMPI and self.icomm:
            # Await that existing scenario is cleaned up
            # self.icomm.send(data, dest=0, tag=1)
            data = self.icomm.recv(source=0, tag=MPI.ANY_TAG)
            print("Existing Scenario has been cleaned up", data)
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
                'reset': False} # reset should be determined from here, from reward computations
        data = self.icomm.sendrecv(data, dest=0, sendtag=2, source=0, recvtag=MPI.ANY_TAG)
        done = data['done']
        state = reward = None
        if not done:
            sensor_data = data['sensor_data']
            criterias = data['criterias']
            velocity = data['velocity']
            reward = self._compute_reward(criterias)
            state = self._process_state(sensor_data, velocity)
            # print(sensor_data.keys())
            # print(sensor_data)
            # print(criterias)
            # save dictionary to pickle file
            with open('state.pickle', 'wb') as fp:
                pickle.dump(state, fp, protocol=pickle.HIGHEST_PROTOCOL)
            with open('trainer_receive.txt', mode='w') as fp:
                fp.writelines(time.asctime()+'\n')
                fp.writelines(f'Done: {str(done)}\n')
                fp.writelines(f'Velocity: {velocity}\n')
        # print(done)

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