#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementation.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import sys
import time

import py_trees
import carla 
import numpy as np

from srunner.autoagents.agent_wrapper import AgentWrapper
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.result_writer import ResultOutputProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from mpi4py import MPI



class ScenarioManager(object):

    """
    Basic scenario manager class. This class holds all functionality
    required to start, and analyze a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. Trigger a result evaluation with manager.analyze_scenario()
    5. If needed, cleanup with manager.stop_scenario()
    """

    def __init__(self, debug_mode=False, sync_mode=False, timeout=2.0, use_mpi=False, is_learner=False):
        """
        Setups up the parameters, which will be filled at load_scenario()

        """
        self.scenario = None
        self.scenario_tree = None
        self.scenario_class = None
        self.ego_vehicles = None
        self.other_actors = None

        self._debug_mode = debug_mode
        self._agent = None
        self._sync_mode = sync_mode
        self._watchdog = None
        self._timeout = timeout
        self.use_mpi = use_mpi
        self.is_learner = is_learner
        if use_mpi:
            self.icomm = MPI.Comm.Get_parent()
            self.size = self.icomm.Get_size()
            self.rank = self.icomm.Get_rank()

        self._running = False
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None

    def _reset(self):
        """
        Reset all parameters
        """
        self._running = False
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        GameTime.restart()

    def cleanup(self):
        """
        This function triggers a proper termination of a scenario
        """

        if self._watchdog is not None:
            self._watchdog.stop()
            self._watchdog = None

        if self.scenario is not None:
            self.scenario.terminate()

        if self._agent is not None:
            self._agent.cleanup()
            self._agent = None

        CarlaDataProvider.cleanup()

    def load_scenario(self, scenario, agent=None):
        """
        Load a new scenario
        """
        self._reset()
        self._agent = AgentWrapper(agent) if agent else None
        if self._agent is not None:
            self._sync_mode = True
        self.scenario_class = scenario
        self.scenario = scenario.scenario
        self.scenario_tree = self.scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

        if self._agent is not None:
            agent = self._agent._agent
            if hasattr(agent, "setup_criterias"):
                agent.setup_criterias(self.scenario.test_criteria)
            self._agent.setup_sensors(self.ego_vehicles[0], self._debug_mode)

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        print("ScenarioManager: Running scenario {}".format(self.scenario_tree.name))
        self.start_system_time = time.time()
        start_game_time = GameTime.get_time()

        self._watchdog = Watchdog(float(self._timeout))
        self._watchdog.start()
        self._running = True

        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                self._tick_scenario(timestamp)

        self.cleanup()

        self.end_system_time = time.time()
        end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - \
            self.start_system_time
        self.scenario_duration_game = end_game_time - start_game_time

        if self.scenario_tree.status == py_trees.common.Status.FAILURE:
            print("ScenarioManager: Terminated due to failure")

    def _tick_scenario(self, timestamp):
        """
        Run next tick of scenario and the agent.
        If running synchornously, it also handles the ticking of the world.
        """
        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()

            if self._debug_mode:
                print("\n--------- Tick ---------\n")

            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()

            if self._agent is not None:
                if self.use_mpi:
                    # with open('receive_action.txt', mode='w') as fp:
                    #     fp.writelines(time.asctime())
                    #     fp.writelines('\nAwaiting message')
                    data = self.icomm.recv(source=0, tag=MPI.ANY_TAG)
                    use_npc = data['use_npc']
                    npc_action = self._agent._agent.run_step(None,None,use_npc=use_npc)  # pylint: disable=not-callable
                    ego_action = carla.VehicleControl(**data['action']) if not use_npc and data['action'] else npc_action
                    if data['reset']:
                        self._running = False     
                else:
                    ego_action = self._agent()  # pylint: disable=not-callable

                self.ego_vehicles[0].apply_control(ego_action)

            # Tick scenario
            self.scenario_tree.tick_once()

            if self._debug_mode:
                print("\n")
                py_trees.display.print_ascii_tree(self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

        if self._sync_mode and self._running and self._watchdog.get_status():
            CarlaDataProvider.get_world().tick()
        
        if not self._running and self.use_mpi: # prepare for clean up
            self.icomm.send({'done': True,
                             'exception': False,
                             'cleaned': True}, dest=0, tag=2)
            # if self.use_mpi:
            #     self.icomm.Disconnect()
            self.use_mpi = False 

        if self._agent is not None and self.use_mpi:
            # get state data 
            vehicle_agent = self._agent._agent
            sensor_data = vehicle_agent.sensor_interface.get_data() 
            velocity = self.get_velocity(self.ego_vehicles[0])
            criterias = self.process_criterias(vehicle_agent.criterias)
            self._agent._agent.run_step(sensor_data,None,use_npc=False) # trigger visualization
            send_data = {
                'sensor_data': sensor_data,
                'criterias': criterias,
                'velocity': velocity,
                'done': not self._running,
                'exception': False, 
                'cleaned': False,
                'npc_act': self.control_to_dict(ego_action) if data['use_npc'] else None 
            }
            self.icomm.send(send_data, dest=0, tag=2)


    def process_criterias(self, criterias):
        data = {}
        for criteria in criterias:
            if hasattr(criteria, 'to_pickable'):
                data[criteria.name] = criteria.to_pickable()
            # if criteria.name == 'RouteCompletionTest':
            #     data['RouteCompletionTest'] = criteria.to_pickable()
            # elif criteria.name == 'InRouteTest':
            #     pass 
        return data 

    def control_to_dict(self, vehicle_control):
        return {
        'steer': vehicle_control.steer,
        'throttle': vehicle_control.throttle,
        'brake': vehicle_control.brake,
        'reverse': vehicle_control.reverse,
        'hand_brake': vehicle_control.hand_brake,
        'manual_gear_shift': vehicle_control.manual_gear_shift,
        'gear': vehicle_control.gear,
        }

    def get_velocity(self, ego_vehicle):
        velocity_vec = ego_vehicle.get_velocity()
        velocity = np.sqrt(velocity_vec.x**2 + velocity_vec.y**2)
        return velocity

    def get_running_status(self):
        """
        returns:
           bool:  False if watchdog exception occured, True otherwise
        """
        return self._watchdog.get_status()

    def stop_scenario(self):
        """
        This function is used by the overall signal handler to terminate the scenario execution
        """
        self._running = False

    def analyze_scenario(self, stdout, filename, junit, json):
        """
        This function is intended to be called from outside and provide
        the final statistics about the scenario (human-readable, in form of a junit
        report, etc.)
        """

        failure = False
        timeout = False
        result = "SUCCESS"

        if self.scenario.test_criteria is None:
            print("Nothing to analyze, this scenario has no criteria")
            return True

        for criterion in self.scenario.get_criteria():
            if (not criterion.optional and
                    criterion.test_status != "SUCCESS" and
                    criterion.test_status != "ACCEPTABLE"):
                failure = True
                result = "FAILURE"
            elif criterion.test_status == "ACCEPTABLE":
                result = "ACCEPTABLE"

        if self.scenario.timeout_node.timeout and not failure:
            timeout = True
            result = "TIMEOUT"

        output = ResultOutputProvider(self, result, stdout, filename, junit, json)
        output.write()

        return failure or timeout
