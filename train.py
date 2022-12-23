from srunner.tools import dotdict
from agent import Trainer, Decay_Explore_Rate
from model import hd_net_args, HDDriveDQN_V0_GRU

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
scenario_args.agent = '.\\srunner\\autoagents\\npc_agent.py' # agent module location
# scenario_args.agent = '.\\srunner\\autoagents\\simple_agent.py' # agent module location
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
scenario_args.numOfWPs = 75 # number of waypoints to draw in front of car
scenario_args.route_dist = 200
scenario_args.raise_excep = False

agent_config = dotdict({
            'sensor_setup': 'hd_map',
            'image_width': '96',
            'image_height': '96',
            'visualize_sensors': '1',
            'external_visualizer': '1',
            'fill_buffer': '0',
            'file': 'test.json',
        })

exp_args = dotdict()
exp_args.BATCH_SZ = 32 # batch size for replay training
exp_args.REPLARY_MEM_SZ = 15_000 # size of the buffer to store (s,a,r,s') tuples
exp_args.NUM_FRAMES = 16  # number of temporal frames the model should use
exp_args.START_STEP = 0 # to restart from previous crash point
exp_args.NUM_STEPS = 3_000_000 # number of steps/frames to train on
exp_args.EXPLORE_STEPS = 250_000 # eps will decay from 1 towards .1 lineary from step 0 -> EXPLORE_STEPS
exp_args.MODEL_NAME = 'hd_gru_8f'
exp_args.TARGET_UPDATE_FREQ = 1000 # frequency in steps/frames to update target model weights 
exp_args.LOAD_MODEL = False
exp_args.TEST = False
exp_args.TEST_EVERY = 10 # test every 10'th episode

model = HDDriveDQN_V0_GRU
model_args = hd_net_args
explore_rate = Decay_Explore_Rate

if __name__ == '__main__':
    try:
        print("Starting training of agent")
        trainer = Trainer(scenario_args, agent_config, model, 
                          model_args, exp_args, explore_rate, debug=False)
        trainer.train()
        scenario_args = trainer.scenario_specifications        
    finally:
        print("Disconnecting")
        trainer.env.comm.Disconnect() 