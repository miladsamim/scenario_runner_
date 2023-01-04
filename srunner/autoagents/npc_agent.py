#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an NPC agent to control the ego vehicle
"""

from __future__ import print_function

import carla
from agents.navigation.basic_agent import BasicAgent

from srunner.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.actorcontrols.visualizer import Visualizer
from srunner.autoagents.sensor_config import SensorConfiguration
from srunner.autoagents.sensor_display import Display


class NpcAgent(AutonomousAgent):

    """
    NPC autonomous agent to control the ego vehicle
    """

    _agent = None
    _route_assigned = False
    opt_dict = {}
    opt_dict['ignore_traffic_lights'] = True

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self._agent = None
        self.hero_actor = None
        with (open(path_to_conf_file, "r")) as f:
            lines = f.read().split("\n")
            self.sensor_setup = lines[0].split(" ")[1]
            self._width = int(lines[1].split(" ")[1])
            self._height = int(lines[2].split(" ")[1])
            self._visualize_sensors = bool(int(lines[3].split(" ")[1]))
            self._external_visualizer = bool(int(lines[4].split(" ")[1]))
            self._fill_buffer = bool(int(lines[5].split(" ")[1]))

        self.sensor_config = SensorConfiguration(self.sensor_setup, self._width, self._height)
        
        if self._visualize_sensors:
            self._display = Display(self._width, self._height, self.sensor_setup)  

    def setup_criterias(self, criterias):
        self.criterias = criterias 

    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]
        """
        return self.sensor_config.get_sensor_configuration()

    def _get_hero_actor(self):
        hero_actor = None
        for actor in CarlaDataProvider.get_world().get_actors():
            if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                hero_actor = actor
                self.visualizer = Visualizer(hero_actor, self.criterias)
                break
        return hero_actor

    def run_step(self, input_data, timestamp, use_npc=True):
        """
        Execute one step of navigation.
        """
        if not self._agent:
           # Search for the ego actor
            if not self.hero_actor:
                self.hero_actor = self._get_hero_actor()
                return carla.VehicleControl()
            # Add an agent that follows the route to the ego
            if self.hero_actor:
                self._agent = BasicAgent(self.hero_actor, 30, opt_dict=self.opt_dict)
                plan = []
                prev_wp = None
                for transform, _ in self._global_plan_world_coord:
                    wp = CarlaDataProvider.get_map().get_waypoint(transform.location)
                    if prev_wp:
                        plan.extend(self._agent.trace_route(prev_wp, wp))
                    prev_wp = wp

                self._agent.set_global_plan(plan)
            else: 
                print("Can not find")
            return carla.VehicleControl()
        else:
            if input_data:
                self._visualization(input_data)
            if use_npc:
                return self._agent.run_step()
            else: 
                return carla.VehicleControl()
    
    def _visualization(self, input_data):
        if self._visualize_sensors:
            self._display.run_interface(input_data)
        if self._external_visualizer and self.hero_actor:
            self.visualizer.render()

    def destroy(self):
        """
        Cleanup
        """
        if self.hero_actor:
            if self._visualize_sensors:
                self._display.quit_interface = True 
            if self._external_visualizer:
                self.visualizer.reset()