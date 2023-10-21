import collections
import math
import time

import carla

client = carla.Client("localhost", 2000)
client.set_timeout(30.0)
world = client.get_world()

# Spawn vehicle
vehicle_bp = world.get_blueprint_library().find("vehicle.tesla.model3")
vehicle = world.spawn_actor(vehicle_bp, world.get_map().get_spawn_points()[0])
vehicle.set_autopilot(True)

SECTIONS = 180
MAX_DISTANCE = 50  # same as radar range
section_data = collections.defaultdict(
    lambda: {"up": MAX_DISTANCE, "down": MAX_DISTANCE}
)


def clamp(min_v, max_v, value):
    return max(min_v, min(value, max_v))


def rad_callback(radar_data):
    velocity_range = 7.5  # m/s
    current_rot = radar_data.transform.rotation

    # Reset section data
    for key in section_data:
        section_data[key]["up"] = MAX_DISTANCE
        section_data[key]["down"] = MAX_DISTANCE
    print(
        max([math.degrees(detect.azimuth) for detect in radar_data]),
        min([math.degrees(detect.azimuth) for detect in radar_data]),
    )
    for detect in radar_data:
        azi = math.degrees(detect.azimuth)
        alt = math.degrees(detect.altitude)
        section = int((azi + 90) % 180)  # +90 to shift from [-90, 90] to [0, 180]

        # Determine if the detection is "up" or "down"
        part = "up" if alt >= 0 else "down"

        # Update the closest point for the section
        section_data[section][part] = min(section_data[section][part], detect.depth)

        # The 0.25 adjusts a bit the distance so the dots can be properly seen
        fw_vec = carla.Vector3D(x=section_data[section][part] - 0.25)
        carla.Transform(
            carla.Location(),
            carla.Rotation(
                pitch=current_rot.pitch + alt,
                yaw=current_rot.yaw + (section - 90),  # Convert back to [-90, 90]
                roll=current_rot.roll,
            ),
        ).transform(fw_vec)

        # Velocity coloring remains the same
        norm_velocity = detect.velocity / velocity_range  # range [-1, 1]
        r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
        g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
        b = int(abs(clamp(-1.0, 0.0, -1.0 - norm_velocity)) * 255.0)

        world.debug.draw_point(
            radar_data.transform.location + fw_vec,
            size=0.075,
            life_time=0.06,
            persistent_lines=False,
            color=carla.Color(r, g, b),
        )


rad_cam = []
rad_bp = world.get_blueprint_library().find("sensor.other.radar")
rad_bp.set_attribute("horizontal_fov", str(30))
rad_bp.set_attribute("vertical_fov", str(20))
rad_bp.set_attribute("range", str(50))
rad_location = carla.Location(x=2.0, z=1.0)
rad_rotation = carla.Rotation(pitch=5, yaw=15)
rad_transform = carla.Transform(rad_location, rad_rotation)
rad_ego = world.spawn_actor(
    rad_bp,
    rad_transform,
    attach_to=vehicle,
    attachment_type=carla.AttachmentType.Rigid,
)
rad_ego.listen(lambda radar_data: rad_callback(radar_data))
rad_cam.append(rad_ego)

try:
    time.sleep(600)
finally:
    rad_ego.destroy()
    vehicle.destroy()
