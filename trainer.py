import traceback

from srunner.tools import dotdict
from scenario_runner import ScenarioRunner


import sys

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
scenario_args.agent = '.\\srunner\\autoagents\\human_agent.py' # agent module location
# scenario_args.agent = '.\\srunner\\autoagents\\npc_agent.py' # agent module location
# scenario_args.agent = '.\\srunner\\autoagents\\simple_agent.py' # agent module location
# scenario_args.agentConfig = '.\\srunner\\autoagents\\simple_agent_config.txt' 
scenario_args.output = False
scenario_args.file = False 
scenario_args.junit = False 
scenario_args.json = False 
scenario_args.outputDir = '' 
scenario_args.configFile = ''
scenario_args.additionalScenario = '' 
scenario_args.debug = True 
scenario_args.reloadWorld = False 
scenario_args.record = '' 
scenario_args.randomize = False 
scenario_args.repetitions = 1
scenario_args.waitForEgo = False 

agent_config = {
            'sensor_setup': 'hd_map',
            'image_width': '500',
            'image_height': '400',
            'visualize_sensors': '1',
            'external_visualizer': '1',
            'file': 'test.json',
        }

class Trainer:
    def __init__(self, scenario_specifications, iterations, debug=False, agentConfig=None):
        self.scenario_specifications = scenario_specifications
        self.iterations = iterations 
        self.scenario_specifications.debug = debug 
        self.agentConfig = agentConfig 

        if agentConfig:
            self._setup_agent()
        #self._setup_scenario()

    def _run_scenario(self):
        scenario_runner = None
        result = True
        try:
            self._setup_scenario()
            scenario_runner = ScenarioRunner(self.scenario_specifications)
            result = scenario_runner.run()
        except Exception:   # pylint: disable=broad-except
            traceback.print_exc()

        finally:
            if scenario_runner is not None:
                print("destroyed")
                scenario_runner.destroy()
                del scenario_runner

    def _setup_scenario(self):
        prev_path = self.scenario_specifications.agentConfig
        self.scenario_specifications.agentConfig = self._agent_config_path if self.agentConfig else prev_path

    def _setup_agent(self):
        model_path = '.\\model_store\\' + agent_config['sensor_setup'] + '_' +\
                     agent_config['image_width'] + 'w_' + agent_config['image_height'] + 'h_'
        agent_config['model_path'] = model_path
        new_path = self.scenario_specifications.agentConfig[:-4] + '_new.txt'
        with open(new_path, 'w') as fp:
            for key, value in agent_config.items(): 
                fp.write('%s: %s\n' % (key, value))
        self._agent_config_path = new_path

    def _prefill_buffer(self):
        pass

    def train(self):
        self._run_scenario() 

    def _monitor_criterias(self):
        pass 
    
if __name__ == '__main__':
    print("Starting training of agent")
    trainer = Trainer(scenario_args, 1, debug=True, agentConfig=None)
    sys.exit(trainer._run_scenario())