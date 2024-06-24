import math
import random
import time

import carla
import numpy as np


def get_waypoints_across_lanes(waypoint):
    """
    Get waypoints across different lanes given an initial waypoint.

    Args:
    - waypoint: The initial CARLA waypoint.

    Returns:
    - A list of waypoints across different lanes.
    """
    waypoints_across_lanes = []

    # Add the initial waypoint to the list
    waypoints_across_lanes.append(waypoint)

    # Get waypoints on the left lanes
    left_waypoint = waypoint.get_left_lane()
    while (
        left_waypoint is not None and left_waypoint.lane_type == carla.LaneType.Driving
    ):
        waypoints_across_lanes.append(left_waypoint)
        left_waypoint = left_waypoint.get_left_lane()

    # Get waypoints on the right lanes
    right_waypoint = waypoint.get_right_lane()
    while (
        right_waypoint is not None
        and right_waypoint.lane_type == carla.LaneType.Driving
    ):
        waypoints_across_lanes.append(right_waypoint)
        right_waypoint = right_waypoint.get_right_lane()

    return waypoints_across_lanes


def draw_waypoints(world, map, life_time=0):
    """
    Draw all waypoints in the world for visualization.

    Args:
    - world: The CARLA world object.
    - map: The CARLA map object.
    - life_time: Duration for which each waypoint mark will be visible (in seconds).
    """
    # Set waypoint parameters
    spacing = 1.0  # distance between waypoints in meters
    waypoints = map.generate_waypoints(spacing)

    # Loop through waypoints and draw them
    for waypoint in waypoints:
        world.debug.draw_point(
            waypoint.transform.location,
            size=0.05,
            color=carla.Color(r=255, g=0, b=0),
            life_time=life_time,
        )


# Connect to the CARLA server
client = carla.Client("localhost", 2000)
client.set_timeout(30.0)
client.load_world("Town04")
world = client.get_world()

# Get the map and its spawn points
map = world.get_map()
# spawn_point = map.get_spawn_points()[346]
# spawn_point.location.x -= 80
# waypoint = map.get_waypoint(spawn_point.location, project_to_road=True)
# waypoints_across_lanes = get_waypoints_across_lanes(waypoint)
# for waypoint in waypoints_across_lanes:
#     world.debug.draw_point(
#         waypoint.transform.location,
#         size=0.05,
#         color=carla.Color(r=0, g=255, b=0),
#         life_time=-1,
#     )
# # world.debug.draw_point(
# #     waypoint.transform.location,
# #     size=0.05,
# #     color=carla.Color(r=0, g=255, b=0),
# #     life_time=-1,
# # )
# # Draw all waypoints
# # draw_waypoints(world, map)
spawn_points = map.get_spawn_points()
sampled_spawn_points = random.sample(spawn_points, len(spawn_points) // 2)

# Spawn NPC vehicles
for sp in sampled_spawn_points:
    # Get a random blueprint.
    blueprint = random.choice(world.get_blueprint_library().filter("vehicle.*"))

    # Some vehicles do not support autopilot, so we need to check and possibly choose again.
    while (
        blueprint.has_attribute("number_of_wheels")
        and int(blueprint.get_attribute("number_of_wheels")) < 4
    ):
        blueprint = random.choice(world.get_blueprint_library().filter("vehicle.*"))

    # Spawn the vehicle
    vehicle = world.try_spawn_actor(blueprint, sp)
    if vehicle:
        vehicle.set_autopilot(True)

time.sleep(600)
