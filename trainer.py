from srunner.tools import dotdict
# from scenario_runner import ScenarioRunner
from carla_env import CarlaEnv
from agent import DQN_Agent
from model import HDDriveDQN, hd_net_args

# import traceback
from collections import deque

scenario_args = dotdict()
scenario_args.host = '127.0.0.1'
scenario_args.port = '2000'
scenario_args.timeout = '10.0'
scenario_args.trafficManagerPort = '8000'
scenario_args.trafficManagerSeed = '0'
scenario_args.sync = True # is required when training
scenario_args.list = False
scenario_args.scenario = 'StraightDriving_1'
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
scenario_args.debug = False 
scenario_args.reloadWorld = True 
scenario_args.record = '' 
scenario_args.randomize = False 
scenario_args.repetitions = 1
scenario_args.waitForEgo = False
scenario_args.is_learner = True  
scenario_args.useMPI = True
scenario_args.drawWaypoints = True

agent_config = dotdict({
            'sensor_setup': 'hd_map',
            'image_width': '96',
            'image_height': '96',
            'visualize_sensors': '1',
            'external_visualizer': '1',
            'fill_buffer': '0',
            'file': 'test.json',
        })

BATCH_SZ = 32 # batch size for replay training
REPLARY_MEM_SZ = 15_000 # size of the buffer to store (s,a,r,s') tuples
NUM_FRAMES = 4  # number of temporal frames the model should use
NUM_EPISODES = 10 # number of episodes to train on 

class Trainer:
    def __init__(self, scenario_specifications, agentConfig, model_args, episodes, debug=False):
        self.scenario_specifications = scenario_specifications
        self.episodes = episodes 
        self.scenario_specifications.debug = debug 
        self.agentConfig = agentConfig
        self.agent = DQN_Agent(HDDriveDQN, model_args, BATCH_SZ, REPLARY_MEM_SZ, NUM_FRAMES)

        self.env = CarlaEnv(scenario_specifications, agentConfig)
        
    def _prefill_buffer(self):
        pass

    def train(self):
        for ep_i in range(NUM_EPISODES):
            state_frame_stack = deque(maxlen=NUM_FRAMES)
            self.env.reset_env()
            if not self.scenario_specifications.useMPI:
                return
            state, reward, done = self.env.step(None)
            for i in range(NUM_FRAMES):
                state_frame_stack.append(state)
            while not done:
                # action = {'throttle':1.0}
                action_idx, action = self.agent.get_split_action(state_frame_stack, epsilon=0.1)
                state, reward, done = self.env.step(action)
                self.agent.add_experience(state, action_idx, reward, done)

                # train when buffer is larger than batch size 
                if self.agent.replay_memory.__len__() > BATCH_SZ:
                    self.agent.experience_replay_split_acts() 

            self.agent.save_model()


    def _monitor_criterias(self):
        pass 
    
if __name__ == '__main__':
    try:
        print("Starting training of agent")
        trainer = Trainer(scenario_args, agent_config, hd_net_args, 1, debug=False)
        trainer.train()
        scenario_args = trainer.scenario_specifications        
    finally:
        print("Disconnecting")
        trainer.env.comm.Disconnect() 
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