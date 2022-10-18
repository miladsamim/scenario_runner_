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
            self._visualize = bool(int(lines[3].split(" ")[1]))
        
        if self._visualize:
            self._display = Display(self._width, self._height, self.sensor_setup)

    def sensors(self):
        """
        Define the sensor suite required by the agent
        """
        img_w = self._width
        img_h = self._height 
        base_sensors = [
                            {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'},
                            {'type': 'sensor.other.imu', 'id': 'IMU'}
                       ]
        sensors_hd_map = base_sensors + [
                            {'id': 'bev_sem','type': 'sensor.camera.semantic_segmentation', 'x':0, 'y': 0, 'z':40, 'pitch':-90, 
                             'yaw': 0, 'roll':0, 'width':img_w, 'height':img_h, 'fov':100},
                         ]

        sensors_hd_map_frontal = sensors_hd_map + [
                                    {'id': 'front_sem', 'type': 'sensor.camera.semantic_segmentation', 'x':.4, 'y': 0, 'z':1.5, 'pitch':0, 
                                     'yaw': 0, 'roll':0, 'width':img_w, 'height':img_h, 'fov':100},
                                    {'id': 'front_right_sem', 'type': 'sensor.camera.semantic_segmentation', 'x':.4, 'y': 0, 'z':1.5, 'pitch':0, 
                                    'yaw': 60, 'roll':0, 'width':img_w, 'height':img_h, 'fov':100},
                                    {'id': 'front_left_sem', 'type': 'sensor.camera.semantic_segmentation', 'x':.4, 'y': 0, 'z':1.5, 'pitch':0, 
                                    'yaw': -60, 'roll':0, 'width':img_w, 'height':img_h, 'fov':100},
                                ]
        full_view = base_sensors + [   
                        {'id': '360_0_sem','type': 'sensor.camera.semantic_segmentation', 'x':0, 'y': 0, 'z':2.2, 'pitch':0, 
                                     'yaw': 0, 'roll':0, 'width':img_w, 'height':img_h, 'fov':100},
                        {'id': '360_1_sem','type': 'sensor.camera.semantic_segmentation', 'x':0, 'y': 0, 'z':2.2, 'pitch':0, 
                            'yaw': 60, 'roll':0, 'width':img_w, 'height':img_h, 'fov':100},
                        {'id': '360_2_sem','type': 'sensor.camera.semantic_segmentation', 'x':0, 'y': 0, 'z':2.2, 'pitch':0, 
                            'yaw': 120, 'roll':0, 'width':img_w, 'height':img_h, 'fov':100}, 
                        {'id': '360_3_sem','type': 'sensor.camera.semantic_segmentation', 'x':0, 'y': 0, 'z':2.2, 'pitch':0, 
                            'yaw': 180, 'roll':0, 'width':img_w, 'height':img_h, 'fov':100},
                        {'id': '360_4_sem','type': 'sensor.camera.semantic_segmentation', 'x':0, 'y': 0, 'z':2.2, 'pitch':0, 
                            'yaw': 240, 'roll':0, 'width':img_w, 'height':img_h, 'fov':100},
                        {'id': '360_5_sem','type': 'sensor.camera.semantic_segmentation', 'x':0, 'y': 0, 'z':2.2, 'pitch':0, 
                            'yaw': 300, 'roll':0, 'width':img_w, 'height':img_h, 'fov':100},
                    ]
        

        if self.sensor_setup == 'hd_map':
            sensors = sensors_hd_map
        elif self.sensor_setup == 'hd_map_frontal':
            sensors = sensors_hd_map_frontal
        elif self.sensor_setup == '360_map':
            sensors = full_view
        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        # print("=====================>")
        # for key, val in input_data.items():
        #     if hasattr(val[1], 'shape'):
        #         shape = val[1].shape
        #         print("[{} -- {:06d}] with shape {}".format(key, val[0], shape))
        #     else:
        #         print("[{} -- {:06d}] ".format(key, val[0]))
        # print("<=====================")
        if self._visualize:
            self._display.run_interface(input_data)

        # DO SOMETHING SMART
        # time.sleep(2)
        # RETURN CONTROL
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.5
        control.brake = 0.0
        control.hand_brake = False

        return control
    
    def destroy(self):
        """
        Cleanup
        """
        if self._visualize:
            self._display.quit_interface = True 


class Display(object):

    """
    Class to control a vehicle manually for debugging purposes
    """

    def __init__(self, width, height, sensor_setup):
        self.use_pygame = False
        self._width = width
        self._height = height
        self._surface = None
        self.setup_sensor_specifics(sensor_setup)
        

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display_name = f"Displaying sensors: {sensor_setup}"
        self._display = pygame.display.set_mode((self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption(self._display_name)

    def setup_sensor_specifics(self, sensor_setup):
        if sensor_setup == 'hd_map':
            self.layout_func = self.hd_map_layout
        elif sensor_setup == 'hd_map_frontal':
            self.layout_func = self.hd_map_frontal_layout
            self._width = self._width * 3 
            self._height = self._height * 2 
        elif sensor_setup == '360_map':
            self.layout_func = self.full_view_layout
            self._width = self._width * 3 
            self._height = self._height * 2 

    def run_interface(self, input_data):
        """
        Run the GUI
        """
        # process sensor data
        # image_center = input_data['bev_sem'][1][:, :, -2::-1]
        layout = self.layout_func(input_data)

        # display image
        self._surface = pygame.surfarray.make_surface(layout.swapaxes(0, 1))
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()
        # to avoid freezes
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return


    def quit_interface(self):
        """
        Stops the pygame window
        """
        pygame.quit()
    
    def hd_map_layout(self, input_data):
        return input_data['bev_sem'][1][:, :, -2::-1]
    
    def hd_map_frontal_layout(self, input_data):
        hd_img = input_data['bev_sem'][1][:, :, -2::-1]
        row1 = np.concatenate([input_data['front_left_sem'][1][:, :, -2::-1], input_data['front_sem'][1][:, :, -2::-1], 
                               input_data['front_right_sem'][1][:, :, -2::-1]], axis=1)
        row2 = np.concatenate([np.zeros_like(hd_img), hd_img, np.zeros_like(hd_img)], axis=1)
        tiled = np.concatenate([row1,row2], axis=0)
        return tiled 
    
    def full_view_layout(self, input_data):
        row1 = np.concatenate([input_data['360_5_sem'][1][:, :, -2::-1], input_data['360_0_sem'][1][:, :, -2::-1], 
                               input_data['360_1_sem'][1][:, :, -2::-1]], axis=1)
        row2 = np.concatenate([input_data['360_4_sem'][1][:, :, -2::-1], input_data['360_3_sem'][1][:, :, -2::-1], 
                               input_data['360_2_sem'][1][:, :, -2::-1]], axis=1)
        tiled = np.concatenate([row1,row2], axis=0) 
        return tiled