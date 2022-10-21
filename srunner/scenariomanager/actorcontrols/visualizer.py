#!/usr/bin/python

# Copyright (c) 2021 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides an OpenCV based visualizer for camera sensors
attached to an actor.

The modul can for example be used inside an OSC Actor controller,
such as simple_vehicle_control.py

It can also be used as blueprint to implement custom visualizers.
"""

import cv2
import numpy as np

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


class Visualizer(object):

    """
    Visualizer class for actor controllers.

    The class provides a birdeye camera and a camera mounted in the front
    bumper of the actor. The resolution is 1000x400 for both RGB cameras.

    To use this class, it is only required to:
    1. Add an instance inside the controller constructor
        visualizer = Visualizer(actor)
    2. Call the render method on a regular basis
        visualizer.render()
    3. Call the reset method during cleanup
        visualizer.reset()

    Args:
        actor (carla.Actor): Vehicle actor the cameras should be attached to.

    Attributes:
        _actor (carla.Actor): The reference actor
        _cv_image_bird (numpy array): OpenCV image for the birdeye camera
        _cv_image_actor (numpy array): OpenCV image for the bumper camera
        _camera_bird (carla.Camera): Birdeye camera
        _camera_actor (carla.Camera): Bumper camera
        _video_writer (boolean): Flag to disable/enable writing the image stream into a video
    """

    _video_writer = False

    def __init__(self, actor, criterias=None):
        self._actor = actor
        self._cv_image_bird = None
        self._cv_image_actor = None
        self._camera_bird = None
        self._camera_actor = None

        bp = CarlaDataProvider.get_world().get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', '650')
        bp.set_attribute('image_size_y', '400')
        self._camera_bird = CarlaDataProvider.get_world().spawn_actor(bp, carla.Transform(
            carla.Location(x=20.0, z=30.0), carla.Rotation(pitch=-90, yaw=-90)), attach_to=self._actor)
        self._camera_bird.listen(lambda image: self._on_camera_update(
            image, birdseye=True))  # pylint: disable=unnecessary-lambda

        bp = CarlaDataProvider.get_world().get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', '650')
        bp.set_attribute('image_size_y', '400')
        self._camera_actor = CarlaDataProvider.get_world().spawn_actor(bp, carla.Transform(
            carla.Location(x=2.3, z=1.0)), attach_to=self._actor)
        self._camera_actor.listen(lambda image: self._on_camera_update(
            image, birdseye=False))  # pylint: disable=unnecessary-lambda

        self.log_window = np.zeros((100,650,3), dtype=np.uint8)

        if self._video_writer:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._video = cv2.VideoWriter('recorded_video.avi', fourcc, 13, (1000, 800))
        
        self.criterias = criterias

    def reset(self):
        """
        Reset cameras
        """
        if self._camera_bird:
            self._camera_bird.destroy()
            self._camera_bird = None
        if self._camera_actor:
            self._camera_actor.destroy()
            self._camera_actor = None

    def _on_camera_update(self, image, birdseye):
        """
        Callback for the camera sensor

        Sets the OpenCV image (_cv_image). Requires conversion from BGRA to RGB.
        """
        if not image:
            return

        image_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        np_image = np.reshape(image_data, (image.height, image.width, 4))
        np_image = np_image[:, :, :3]
        np_image = np_image[:, :, ::-1]
        if not birdseye:
            self._cv_image_actor = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        else:
            self._cv_image_bird = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

    criteria_text_generic = {
        'RouteCompletionTest': 'Route Completion: ',
        'CollisionTest': 'Collision: ',
        'InRouteTest': 'On Route: ',
        'RunningRedLightTest': 'Running a Red Light: ',
        'RunningStopTest': 'Running a Stop Sign: ',
        'ActorSpeedAboveThresholdTest': 'Actor Blocked: ',
        'OutsideRouteLanesTest': 'Outside Lanes: ',
        'InRouteTest': 'Route Deviation: ',
        'OffRoadTest': 'Off-road: ',
    }

    def render(self):
        """
        Render images in an OpenCV window (has to be called on a regular basis)
        """
        if self._cv_image_actor is not None and self._cv_image_bird is not None:
            # print(type(self._cv_image_actor))
            # print(self.log_window.type(), self._cv_image_actor.type())
            im_v = cv2.vconcat([self.log_window, self._cv_image_actor, self._cv_image_bird])
            cv2.circle(im_v, (600, 400), 40, (170, 170, 170), -1)
            text = str(int(round((self._actor.get_velocity().x * 3.6))))+" kph"

            speed = np.sqrt(self._actor.get_velocity().x**2 + self._actor.get_velocity().y**2)

            text = str(int(round((speed * 3.6))))+" kph"
            text = ' '*(7-len(text)) + text
            im_v = cv2.putText(im_v, text, (565, 405), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 2, cv2.LINE_AA)
            font_scale, font_thickness =.3, 1
            if self.criterias:
                for idx, criteria in enumerate(self.criterias):
                    text = ''
                    base_text = self.criteria_text_generic[criteria.name]
                    if criteria.name == 'RouteCompletionTest':
                        text = base_text + self._get_route_completion_text(criteria) 
                    elif criteria.name == 'InRouteTest':
                        text = base_text + self.get_in_route_text(criteria)
                    elif criteria.name == 'CollisionTest':
                        text = base_text + self._get_collision_text(criteria)
                    elif criteria.name == 'RunningRedLightTest':
                        text = base_text + self.get_red_light_text(criteria)
                    elif criteria.name == 'RunningStopTest':
                        text = base_text + self.get_stop_sign_text(criteria)
                    elif criteria.name == 'ActorSpeedAboveThresholdTest':
                        text = base_text + self.get_actor_speed_limit_text(criteria)
                    elif criteria.name == 'OutsideRouteLanesTest':
                        text = base_text + self.get_outside_route_lane_text(criteria)
                    elif criteria.name == 'OffRoadTest':
                        text = base_text + self.get_off_road_text(criteria)
                    if text:
                        self._visualize_generic(im_v, criteria, idx, text, font_scale, font_thickness)

            cv2.imshow("", im_v)
            cv2.waitKey(1)
            if self._video_writer:
                self._video.write(im_v)

    def get_off_road_text(self, criteria):
        if criteria._offroad:
            return f'Off-road detected for {criteria._time_offroad:.1f} sec.'
        else:
            return ''

    def get_in_route_text(self, criteria):
        if criteria.test_status == 'FAILURE':
            return criteria.list_traffic_events[-1].get_message()
        else:
            return ''

    def get_outside_route_lane_text(self, criteria):
        if criteria._outside_lane_active and criteria._wrong_lane_active:
            return  'Outside driving lanes + At the wrong lane'
        elif criteria._outside_lane_active:
            return  'Outside driving lanes'
        elif criteria._wrong_lane_active:
            return 'At the wrong lane'
        return ''


    def get_actor_speed_limit_text(self, criteria):
        if criteria.test_status == 'FAILURE':
            return criteria.list_traffic_events[-1].get_message()
        else: 
            return f'Not Blocked for {criteria._below_threshold_max_time} sec. yet.'

    def get_stop_sign_text(self, criteria):
        if criteria.test_status == 'FAILURE':
            return  f'Count: {criteria.actual_value}' + ' | New: ' + criteria.list_traffic_events[-1].get_message()
        else:
            return ''

    def get_red_light_text(self, criteria):
        if criteria.test_status == 'FAILURE':
            return  f'Count: {criteria.actual_value}' + ' | New: ' + criteria.list_traffic_events[-1].get_message()
        else:
            return ''

    def _get_route_completion_text(self, criteria):
        if criteria.list_traffic_events:
            return  criteria.list_traffic_events[-1].get_message()
        else:
            return ''

    def _get_collision_text(self, criteria):
        if criteria.test_status == 'FAILURE':
            return  f'Count: {criteria.actual_value}' + ' | New: ' + criteria.list_traffic_events[-1].get_message()
        else:
            return '' 
    def _visualize_generic(self, im_v, criteria, idx, text, font_scale, font_thickness):
        y_offset = idx*10+10
        cv2.putText(im_v, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)