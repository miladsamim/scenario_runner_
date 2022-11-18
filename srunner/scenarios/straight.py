"""
    Scenario to drive straight
"""

import random
import py_trees
import carla

from agents.navigation.local_planner import RoadOption

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.tools.route_manipulation import interpolate_trajectory
from srunner.scenarios.route_scenario import convert_transform_to_location
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      StopVehicle,
                                                                      LaneChange,
                                                                      WaypointFollower,
                                                                      Idle)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import (CollisionTest,
                                                                     InRouteTest,
                                                                     RouteCompletionTest,
                                                                     OutsideRouteLanesTest,
                                                                     RunningRedLightTest,
                                                                     RunningStopTest,
                                                                     ActorSpeedAboveThresholdTest,
                                                                     OffRoadTest)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerDistanceToVehicle, StandStill, InTriggerDistanceToLocation
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_waypoint_in_distance

SECONDS_GIVEN_PER_METERS = 2 


class StraightDriving(BasicScenario):
    """"""

    timeout = 120
    
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=600):
        self.timeout = timeout
        self._map = CarlaDataProvider.get_map()
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self._distance = 400
        
        self._update_route(world, config, debug_mode)
        
        super(StraightDriving, self).__init__("Straight",
                                        ego_vehicles,
                                        config,
                                        world,
                                        debug_mode,
                                        criteria_enable=criteria_enable)
    
    # Agent Scenario Tool
    def _update_route(self, world, config, debug_mode):
        start_location = self._reference_waypoint.transform.location
        self.end_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._distance)
        end_location = self.end_waypoint.transform.location
        gps_route, route = self._get_routes(start_location, end_location)
        
        self.route = route 
        CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(self.route))

        if config.agent: 
            config.agent.set_global_plan(gps_route, self.route)            
        
        self.timeout = self._estimate_route_timeout()

        # Print route in debug mode
        if debug_mode:
            self._draw_waypoints(world, self.route, vertical_shift=0.1, persistency=50000.0)

    # Agent Scenario Tool
    def _estimate_route_timeout(self):
        """
        Estimate the duration of the route
        """
        route_length = 0.0  # in meters

        prev_point = self.route[0][0]
        for current_point, _ in self.route[1:]:
            dist = current_point.location.distance(prev_point.location)
            route_length += dist
            prev_point = current_point

        return int(SECONDS_GIVEN_PER_METERS * route_length)

    # Agent Scenario Tool
    def _get_routes(self, start_loc, end_location, hop_resolution=1.0):    
        # grp = GlobalRoutePlanner(CarlaDataProvider.get_map(), hop_resolution)
        # route = grp.trace_route(start_loc, end_location)
        # trajectory = get_trajectory_from_route(route)
        gps_route, route = interpolate_trajectory([start_loc, end_location], hop_resolution=hop_resolution)
        return gps_route, route 
    
    # Agent Scenario Tool
    def _draw_waypoints(self, world, waypoints, vertical_shift, persistency=-1):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        """
        for w in waypoints:
            wp = w[0].location + carla.Location(z=vertical_shift)

            if w[1] == RoadOption.LEFT:  # Yellow
                color = carla.Color(255, 255, 0)
            elif w[1] == RoadOption.RIGHT:  # Cyan
                color = carla.Color(0, 255, 255)
            elif w[1] == RoadOption.CHANGELANELEFT:  # Orange
                color = carla.Color(255, 64, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT:  # Dark Cyan
                color = carla.Color(0, 64, 255)
            elif w[1] == RoadOption.STRAIGHT:  # Gray
                color = carla.Color(128, 128, 128)
            else:  # LANEFOLLOW
                color = carla.Color(0, 255, 0)  # Green

            world.debug.draw_point(wp, size=0.1, color=color, life_time=persistency)

        world.debug.draw_point(waypoints[0][0].location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(0, 0, 255), life_time=persistency)
        world.debug.draw_point(waypoints[-1][0].location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(255, 0, 0), life_time=persistency)


    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        end_waypoint = self.end_waypoint#, _ = get_waypoint_in_distance(self._reference_waypoint, self._distance)
        end_transform = end_waypoint.transform
        end_transform.location.z += 0.5
        ego_vehicle = CarlaDataProvider.request_new_actor('vehicle.tesla.model3', end_transform)
        self.other_actors.append(ego_vehicle)

    def _create_behavior(self):
        """
        The scenario defined after is a "follow leading vehicle" scenario. After
        invoking this scenario, it will wait for the user controlled vehicle to
        enter the start region, then make the other actor to drive until reaching
        the next intersection. Finally, the user-controlled vehicle has to be close
        enough to the other actor to end the scenario.
        If this does not happen within 60 seconds, a timeout stops the scenario
        """

        # Ego vehicle must drive unto end point 
        end_waypoint = self.end_waypoint#, _ = get_waypoint_in_distance(self._reference_waypoint, self._distance)
        end_transform = end_waypoint.transform
        end_transform.location.z += 0.5

        # end condition
        endcondition = py_trees.composites.Parallel("Waiting for end position",
                                                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        endcondition_part1 = InTriggerDistanceToLocation(actor=self.ego_vehicles[0],
                                                         target_location=end_transform.location,
                                                         distance=10,
                                                         name="FinalDistanceToEndPoint")
        endcondition.add_child(endcondition_part1)

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        sequence.add_child(endcondition)

        return sequence


    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        route = convert_transform_to_location(self.route)

        collision_criterion = CollisionTest(self.ego_vehicles[0], terminate_on_failure=False)

        route_criterion = InRouteTest(self.ego_vehicles[0],
                                      route=route,
                                      offroad_max=30,
                                      terminate_on_failure=False)

        completion_criterion = RouteCompletionTest(self.ego_vehicles[0], route=route)

        outsidelane_criterion = OutsideRouteLanesTest(self.ego_vehicles[0], 
                                                      route, 
                                                      terminate_on_failure=False)

        red_light_criterion = RunningRedLightTest(self.ego_vehicles[0])

        off_road_test = OffRoadTest(self.ego_vehicles[0])

        stop_criterion = RunningStopTest(self.ego_vehicles[0])

        blocked_criterion = ActorSpeedAboveThresholdTest(self.ego_vehicles[0],
                                                         speed_threshold=0.1,
                                                         below_threshold_max_time=10.0,
                                                         terminate_on_failure=False)

        criteria.append(completion_criterion)
        criteria.append(route_criterion)
        criteria.append(outsidelane_criterion)
        criteria.append(off_road_test)
        criteria.append(blocked_criterion)
        criteria.append(collision_criterion)
        criteria.append(red_light_criterion)
        criteria.append(stop_criterion)

        return criteria
    
    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()