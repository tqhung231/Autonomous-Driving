import carla
import time

# Connect to the CARLA server
host = "127.0.0.1"
port = 2000
client = carla.Client(host, port)
client.set_timeout(30.0)
client.load_world("Town04")

# Get the world
world = client.get_world()

# Get spawn points
spawn_points = world.get_map().get_spawn_points()

# Visualize spawn points using Debug Helper
debug_helper = world.debug

# Draw small red spheres at each spawn point for 60 seconds
for spawn_point in spawn_points:
    location = spawn_point.location
    debug_helper.draw_point(
        location, size=0.1, color=carla.Color(0, 255, 0), life_time=60.0
    )

# Wait for a while to visualize the points in the simulator
time.sleep(60)
