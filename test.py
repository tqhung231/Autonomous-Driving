import carla
import time

# Connect to the CARLA server
host = "127.0.0.1"
port = 2000
client = carla.Client(host, port)
client.set_timeout(30.0)
client.load_world("Town04")

# Get the world and map
world = client.get_world()
map = world.get_map()

# Get spawn points
spawn_points = world.get_map().get_spawn_points()
spawn_location = (spawn_points[346].location + spawn_points[347].location) / 2
new_spawn_point = carla.Transform(spawn_location, spawn_points[346].rotation)

# # Get waypoint
# waypoint = map.get_waypoint(
#     spawn_point.location,
#     project_to_road=True,
#     lane_type=carla.LaneType.Driving,
# )

# # Visualize spawn points using Debug Helper
# debug_helper = world.debug

# # Draw small red spheres at each spawn point for 60 seconds
# location = spawn_point.location
# debug_helper.draw_point(
#     waypoint, size=0.1, color=carla.Color(0, 255, 0), life_time=60.0
# )

# # location = spawn_point.location + carla.Location(x=60)
# # debug_helper.draw_point(
# #     location, size=0.1, color=carla.Color(0, 255, 0), life_time=60.0
# # )

# # Wait for a while to visualize the points in the simulator
# time.sleep(60)

# Switch to synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05  # Corresponds to the time.sleep value you wanted
world.apply_settings(settings)

# Spawn a vehicle at the new spawn point
blueprint_library = world.get_blueprint_library()
vehicle_blueprint = blueprint_library.filter("model3")[0]
vehicle = world.spawn_actor(vehicle_blueprint, new_spawn_point)

# Attach a lane invasion sensor to the vehicle
sensor_blueprint = blueprint_library.find("sensor.other.lane_invasion")
sensor = world.spawn_actor(sensor_blueprint, carla.Transform(), attach_to=vehicle)

lane_blueprint = blueprint_library.find("sensor.other.collision")
lane = world.spawn_actor(lane_blueprint, carla.Transform(), attach_to=vehicle)


# Callback function for the lane invasion sensor
def on_lane_invasion(event):
    print(f"Lane invaded at: {event}")


def collision(event):
    print(f"Collision at: {event}")


sensor.listen(on_lane_invasion)
lane.listen(collision)

# Set the vehicle to drive straight with a fixed throttle
vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))

# Run in synchronous mode
try:
    while True:
        world.tick()
        time.sleep(0.2)
finally:
    # Cleanup
    sensor.stop()
    sensor.destroy()
    vehicle.destroy()

    # Switch back to asynchronous mode before exiting
    settings.synchronous_mode = False
    world.apply_settings(settings)
