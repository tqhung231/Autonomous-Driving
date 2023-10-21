import math
import threading
import time

import carla
import matplotlib.pyplot as plt
import numpy as np

# Create an event for each flag
events = [threading.Event() for _ in range(6)]

client = carla.Client("localhost", 2000)
client.set_timeout(30.0)
world = client.get_world()

# Enable synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Spawn vehicle
vehicle_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
# vehicle.set_autopilot(True)

# Define the distance you want between the two vehicles
distance = 5  # for example, 5 meters

# Get the current rotation of the first vehicle
rotation = spawn_point.rotation

# Calculate the new spawn point for the second vehicle to make it lie horizontally in front of the first one
new_location = spawn_point.location + carla.Location(x=distance)

# Adjust the rotation of the second vehicle to make it horizontal with respect to the first one
new_rotation = carla.Rotation(
    pitch=rotation.pitch, yaw=rotation.yaw + 90, roll=rotation.roll
)

new_spawn_point = carla.Transform(location=new_location, rotation=new_rotation)

# Spawn the second vehicle
vehicle2 = world.spawn_actor(vehicle_bp, new_spawn_point)


SECTIONS = 180
MAX_DISTANCE = 50  # same as radar range

radar_data_dict = {}


def rad_callback(radar_data, idx):
    radar_data_dict[idx] = radar_data
    events[idx].set()


# Attach radars with their own callbacks
rad_cam = []
for i in range(6):
    rad_bp = world.get_blueprint_library().find("sensor.other.radar")
    rad_bp.set_attribute("horizontal_fov", str(30))
    rad_bp.set_attribute("vertical_fov", str(20))
    rad_bp.set_attribute("range", str(50))
    rad_location = carla.Location(x=2.0, z=1.0)
    rad_rotation = carla.Rotation(pitch=5, yaw=(-75 + 30 * i))
    rad_transform = carla.Transform(rad_location, rad_rotation)
    rad_ego = world.spawn_actor(
        rad_bp,
        rad_transform,
        attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid,
    )
    rad_ego.listen(lambda radar_data, idx=i: rad_callback(radar_data, idx))
    rad_cam.append(rad_ego)

# world.tick()

try:
    for _ in range(100000000):
        world.tick()
        # Wait for all events to be set
        for event in events:
            event.wait()

        combined_data = np.full((SECTIONS, 2), MAX_DISTANCE)

        for idx, radar_data in radar_data_dict.items():
            for detect in radar_data:
                azi = math.degrees(detect.azimuth)
                alt = math.degrees(detect.altitude)
                section = int((azi + 15) % 30)  # +15 to shift from [-15, 15] to [0, 30]
                section = section + 30 * idx
                part_idx = 0 if alt >= 0 else 1  # 0 for up, 1 for down
                combined_data[section, part_idx] = min(
                    combined_data[section, part_idx], detect.depth
                )

        print(combined_data)

        for event in events:
            event.clear()

        # x_values = np.arange(0, 180)
        # y_values_up = combined_data[:, 0]
        # y_values_down = combined_data[:, 1]

        # # Create a scatter plot
        # plt.figure(figsize=(10, 5))
        # plt.scatter(x_values, y_values_up, c="blue", label="Up", s=5)
        # plt.scatter(x_values, y_values_down, c="red", label="Down", s=5)
        # plt.title("Radar Data (Up & Down)")
        # plt.xlabel("Azimuth Angle Sections")
        # plt.ylabel("Distance from Vehicle (m)")
        # plt.xticks(np.arange(0, 181, 10))
        # plt.yticks(np.arange(0, 51, 10))
        # plt.xlim(0, 180)
        # plt.ylim(0, 50)
        # plt.grid(True)
        # plt.legend()

        # plt.show()

finally:
    for rad_ego in rad_cam:
        rad_ego.destroy()
    vehicle.destroy()
    settings.synchronous_mode = False
    world.apply_settings(settings)
