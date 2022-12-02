#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a dummy agent to control the ego vehicle
"""

from __future__ import print_function

import pygame
import cv2
import numpy as np

import carla
import time

from srunner.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenariomanager.actorcontrols.visualizer import Visualizer
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.autoagents.sensor_config import SensorConfiguration
from srunner.autoagents.sensor_display import Display


class SimpleAgent(AutonomousAgent):

    """
    Dummy autonomous agent to control the ego vehicle
    """

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        assert path_to_conf_file is not None 
        with (open(path_to_conf_file, "r")) as f:
            lines = f.read().split("\n")
            self.sensor_setup = lines[0].split(" ")[1]
            self._width = int(lines[1].split(" ")[1])
            self._height = int(lines[2].split(" ")[1])
            self._visualize_sensors = bool(int(lines[3].split(" ")[1]))
            self._external_visualizer = bool(int(lines[4].split(" ")[1]))

        self.sensor_config = SensorConfiguration(self.sensor_setup, self._width, self._height)

        self.hero_actor = None            

        if self._visualize_sensors:
            self._display = Display(self._width, self._height, self.sensor_setup)            

    def _get_hero_actor(self):
        hero_actor = None
        for actor in CarlaDataProvider.get_world().get_actors():
            if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                hero_actor = actor
                if self._external_visualizer:
                    self.visualizer = Visualizer(hero_actor, self.criterias)
                break
        return hero_actor

    def setup_criterias(self, criterias):
        self.criterias = criterias

    def sensors(self):
        """
        Define the sensor suite required by the agent
        """
        return self.sensor_config.get_sensor_configuration()

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        # Search for the ego actor
        if not self.hero_actor:
            self.hero_actor = self._get_hero_actor()
            return carla.VehicleControl()

        # print("=====================>")
        # for key, val in input_data.items():
        #     if hasattr(val[1], 'shape'):
        #         shape = val[1].shape
        #         print("[{} -- {:06d}] with shape {}".format(key, val[0], shape))
        #     else:
        #         print("[{} -- {:06d}] ".format(key, val[0]))
        # print("<=====================")
        self._visualization(input_data)

        # DO SOMETHING SMART
        # time.sleep(2)
        # RETURN CONTROL
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.5
        control.brake = 0.0
        control.hand_brake = False

        return control
    
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