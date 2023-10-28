import math
import random
import threading
import time
from datetime import datetime

import carla
import gymnasium
import numpy as np
import pygame
from gymnasium import spaces

sem = threading.Semaphore(1)
events = [threading.Event() for _ in range(6)]

SECTIONS = 180
MAX_DISTANCE = 50  # same as radar range

DELTA_SECONDS = 0.05


class CarlaEnvContinuous(gymnasium.Env):
    metadata = {"render_modes": ["true", "false"], "render_fps": 20}

    def __init__(self, debug=False, render_mode=True, num_npc=20):
        # Connect to the server
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(30.0)
        self.client.load_world("Town04")

        # Get the world
        self.world = self.client.get_world()

        # Initialize Traffic Manager
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_synchronous_mode(True)

        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DELTA_SECONDS  # FPS
        settings.no_rendering_mode = not render_mode
        self.world.apply_settings(settings)
        self.fixed_delta_seconds = DELTA_SECONDS

        # Set debug mode
        self.debug = debug

        # Set image size
        self.img_width = 640
        self.img_height = 480

        # Additional information
        self.frame = 0
        self.actors = []
        self.img_captured = None
        self.collision = None
        self.lane_invasion = None
        self.lidar_data = None
        self.radar_data_dict = {}
        self.npc = []

        # Set actors
        self.blueprint_library = self.world.get_blueprint_library()

        # Set ego vehicle spawn point
        spawn_points = self.world.get_map().get_spawn_points()
        # chosen_spawn_point = random.choice(spawn_points[345:353])
        self.chosen_spawn_point = spawn_points[347]

        # Setup ego vehicle
        vehicle_bp = self.blueprint_library.find("vehicle.tesla.model3")
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, self.chosen_spawn_point)

        # Spawn NPC vehicles
        self.num_npc = num_npc
        for _ in range(self.num_npc):
            npc = self._spawn_npc()
            self.npc.append(npc)

        # # Setup RGB camera
        # camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        # camera_bp.set_attribute("image_size_x", f"{self.img_width}")
        # camera_bp.set_attribute("image_size_y", f"{self.img_height}")
        # camera_bp.set_attribute("fov", "100")
        # camera_transform = carla.Transform(carla.Location(x=1.3, z=2.3))
        # self.camera = self.world.spawn_actor(
        #     camera_bp, camera_transform, attach_to=self.ego_vehicle
        # )
        # self.camera.listen(self._process_image)
        self.camera = None

        # Setup collision and lane invasion sensors
        collision_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.ego_vehicle
        )
        self.collision_sensor.listen(self._on_collision)

        lane_invasion_bp = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_invasion_sensor = self.world.spawn_actor(
            lane_invasion_bp, carla.Transform(), attach_to=self.ego_vehicle
        )
        self.lane_invasion_sensor.listen(self._on_lane_invasion)

        # # Attach LiDAR sensor to the ego vehicle
        # lidar_bp = self.blueprint_library.find("sensor.lidar.ray_cast")
        # # lidar_bp.set_attribute('range', '100')  # 100 meters range
        # # lidar_bp.set_attribute("channels", "32")  # 32 channels
        # # lidar_bp.set_attribute("rotation_frequency", "10")  # 10 Hz
        # # lidar_bp.set_attribute("upper_fov", "0")  # Upper field of view
        # # lidar_bp.set_attribute("lower_fov", "-30")  # Lower field of view
        # # lidar_bp.set_attribute("points_per_second", "100000")
        # lidar_bp.set_attribute("dropoff_general_rate", "0.0")  # No points dropoff
        # lidar_bp.set_attribute("dropoff_intensity_limit", "0.0")
        # lidar_bp.set_attribute("dropoff_zero_intensity", "0.0")
        # lidar_bp.set_attribute("horizontal_fov", "180.0")
        # self.lidar_sensor = self.world.spawn_actor(
        #     lidar_bp,
        #     carla.Transform(carla.Location(x=0, z=2.4)),
        #     attach_to=self.ego_vehicle,
        # )
        # self.lidar_sensor.listen(self._process_lidar)
        self.lidar_sensor = None

        # Attach radars
        self.rad_cam = []
        for i in range(6):
            rad_bp = self.world.get_blueprint_library().find("sensor.other.radar")
            rad_bp.set_attribute("horizontal_fov", str(30))
            rad_bp.set_attribute("vertical_fov", str(20))
            rad_bp.set_attribute("range", str(50))
            rad_location = carla.Location(x=2.0, z=1.0)
            rad_rotation = carla.Rotation(pitch=5, yaw=(-75 + 30 * i))
            rad_transform = carla.Transform(rad_location, rad_rotation)
            rad_ego = self.world.spawn_actor(
                rad_bp,
                rad_transform,
                attach_to=self.ego_vehicle,
                attachment_type=carla.AttachmentType.Rigid,
            )
            rad_ego.listen(
                lambda radar_data, idx=i: self._process_radar(radar_data, idx)
            )
            self.rad_cam.append(rad_ego)

        # Tick the world
        self.world.tick()

        # Save all actors
        self.actors = [
            self.ego_vehicle,
            self.camera,
            self.collision_sensor,
            self.lane_invasion_sensor,
            self.lidar_sensor,
        ]
        self.actors.extend(self.rad_cam)

        # If in debug mode, enable autopilot
        if self.debug:
            if self.ego_vehicle is not None:
                self.ego_vehicle.set_autopilot(True)

        # # Observations are dictionaries with the sensor data
        # self.observation_space = spaces.Dict(
        #     {
        #         "rbg": spaces.Box(
        #             0, 255, shape=(self.img_height, self.img_width, 3), dtype=np.uint8
        #         ),
        #     }
        # )

        # Observations are dictionaries with the sensor data
        # self.observation_space = spaces.Box(
        #     0, 255, shape=(self.img_height, self.img_width, 3), dtype=np.uint8
        # )
        self.observation_space = spaces.Box(0, 1, shape=(360,), dtype=np.float32)

        # Action corresponding to throtle, steer, brake
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            shape=(2,),
            dtype=np.float32,
        )

        self._follow_agent()

    def reset(self, seed=None):
        # Get the current time
        now = datetime.now()
        with open("log.txt", "a") as file:
            file.write(f"Reset environment at {now.hour}:{now.minute}\n")
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Destroy NPC vehicles
        for npc in self.npc:
            npc.destroy()
        self.npc = []

        self.ego_vehicle.set_transform(self.chosen_spawn_point)

        # Spawn NPC vehicles
        for _ in range(self.num_npc):
            npc = self._spawn_npc()
            self.npc.append(npc)

        # Reset the data
        self.frame = 0
        self.images = []
        self.img_captured = None
        self.collision = None
        self.lane_invasion = None
        self.lidar_data = None
        self.radar_data_dict = {}

        # Tick the world
        self.world.tick()

        # # Wait for image to be captured
        # while self.img_captured is None:
        #     pass

        # # Wait for lidar data
        # while self.lidar_data is None:
        #     pass

        # Get initial observation
        observation = self._get_obs()

        return observation, {}

    def step(self, action):
        print(action)
        # self._follow_agent()
        # Set vehicle control
        if action[0] > 0:
            control = carla.VehicleControl(
                throttle=float(action[0]), steer=float(action[1]), brake=0.0
            )
        else:
            control = carla.VehicleControl(
                throttle=0.0, steer=float(action[1]), brake=float(-action[0])
            )
        self.ego_vehicle.apply_control(control)

        # Tick the world
        self._tick()

        # Calculate the reward
        terminated = False
        truncated = False

        if self.collision:
            terminated = True
            reward = -10.0
            self.collision = None
            return self._get_obs(), reward, terminated, truncated, {}

        reward = float(action[0] - abs(action[1]))
        # if self.lane_invasion:
        #     print("Lane invasion")
        #     reward -= 1
        #     self.lane_invasion = None

        if self.frame >= 1000:
            truncated = True
            return self._get_obs(), reward, terminated, truncated, {}

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        # with sem:
        #     return self.img_captured
        # return self.lidar_data
        for event in events:
            event.wait()

        combined_data = np.full((SECTIONS, 2), MAX_DISTANCE)

        for idx, radar_data in self.radar_data_dict.items():
            for detect in radar_data:
                azi = math.degrees(detect.azimuth)
                alt = math.degrees(detect.altitude)
                section = int((azi + 15) % 30)  # +15 to shift from [-15, 15] to [0, 30]
                section = section + 30 * idx
                part_idx = 1 if alt >= 0 else 0  # 1 for up, 0 for down
                combined_data[section, part_idx] = min(
                    combined_data[section, part_idx], detect.depth / MAX_DISTANCE
                )

        for event in events:
            event.clear()

        return combined_data.flatten()

    def _tick(self):
        self.world.tick()
        self.frame += 1
        # print(f"Time: {self.frame*self.fixed_delta_seconds}s")

    def _process_image(self, image):
        with sem:
            self.img_captured = np.array(image.raw_data, dtype=np.dtype("uint8"))
            self.img_captured = np.reshape(
                self.img_captured, (self.img_height, self.img_width, 4)
            )
            self.img_captured = self.img_captured[:, :, :3]
            self.img_captured = self.img_captured[:, :, ::-1]

    def _process_lidar(self, data):
        with sem:
            # Convert the raw data to numpy array
            self.lidar_data = np.array(data.raw_data, dtype=np.dtype("f4"))
            self.lidar_data = np.reshape(
                self.lidar_data, (int(self.lidar_data.shape[0] / 4), 4)
            )

            self.lidar_data = self.lidar_data[:, :3]

    def _process_radar(self, data, idx):
        self.radar_data_dict[idx] = data
        events[idx].set()

    def _on_collision(self, event):
        self.collision = event

    def _on_lane_invasion(self, event):
        self.lane_invasion = event

    def _follow_agent(self):
        # Get the spectator from the world
        spectator = self.world.get_spectator()

        # Get the car's current transform
        car_transform = self.ego_vehicle.get_transform()

        # Modify the transform to move the spectator
        car_transform.location.z = 50
        car_transform.rotation.pitch = -70

        # Set the spectator's transform
        spectator.set_transform(car_transform)
        # spectator.set_transform(self.camera.get_transform())

    def _spawn_npc(self):
        # Get a random blueprint.
        blueprint = random.choice(
            self.world.get_blueprint_library().filter("vehicle.*")
        )

        # Some vehicles do not support autopilot, so we need to check and possibly choose again.
        while (
            blueprint.has_attribute("number_of_wheels")
            and int(blueprint.get_attribute("number_of_wheels")) < 4
        ):
            blueprint = random.choice(
                self.world.get_blueprint_library().filter("vehicle.*")
            )

        # Define a random color for the vehicle
        if blueprint.has_attribute("color"):
            color = random.choice(blueprint.get_attribute("color").recommended_values)
            blueprint.set_attribute("color", color)

        # Spawn the vehicle
        vehicle = None
        while vehicle is None:
            try:
                spawn_point = random.choice(self.world.get_map().get_spawn_points())
                vehicle = self.world.spawn_actor(blueprint, spawn_point)
            except RuntimeError:
                pass
        vehicle.set_autopilot(True)

        return vehicle

    def _destroy(self):
        for actor in self.actors:
            if actor is not None:
                if isinstance(actor, carla.Sensor):
                    actor.stop()
                    actor.destroy()
                else:
                    actor.destroy()
        self.actors = []

        for npc in self.npc:
            npc.destroy()

        self.npc = []

    def close(self):
        # # Save images and controls
        # data = np.array(self.images)
        # np.save("images.npy", data)
        # # np.savez_compressed('images.npz', array=data)
        # data = np.array(self.controls)
        # np.save("controls.npy", data)
        # # np.savez_compressed('controls.npz', array=data)

        # Ensure synchronous mode is turned off
        self.traffic_manager.set_synchronous_mode(False)
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

        self._destroy()


# class CarlaEnvDiscrete(gymnasium.Env):
#     metadata = {"render_modes": ["true", "false"], "render_fps": 20}

#     def __init__(self, debug=False, render_mode=True):
#         # Connect to the server
#         self.client = carla.Client("127.0.0.1", 2000)
#         self.client.set_timeout(30.0)
#         self.client.load_world("Town04")

#         # Get the world
#         self.world = self.client.get_world()

#         # Set synchronous mode
#         settings = self.world.get_settings()
#         settings.synchronous_mode = True
#         settings.fixed_delta_seconds = DELTA_SECONDS  # FPS
#         settings.no_rendering_mode = not render_mode
#         self.world.apply_settings(settings)
#         self.fixed_delta_seconds = DELTA_SECONDS

#         # Set debug mode
#         self.debug = debug

#         # Set image size
#         self.img_width = 640
#         self.img_height = 480

#         # Additional information
#         self.frame = 0
#         self.actors = []
#         self.img_captured = None
#         self.collision = None
#         self.lane_invasion = None

#         # Set actors
#         self.blueprint_library = self.world.get_blueprint_library()

#         # Set ego vehicle spawn point
#         spawn_points = self.world.get_map().get_spawn_points()
#         # chosen_spawn_point = random.choice(spawn_points[345:353])
#         self.chosen_spawn_point = spawn_points[347]

#         # Setup ego vehicle
#         vehicle_bp = self.blueprint_library.find("vehicle.tesla.model3")
#         self.ego_vehicle = self.world.spawn_actor(vehicle_bp, self.chosen_spawn_point)

#         # Setup RGB camera
#         camera_bp = self.blueprint_library.find("sensor.camera.rgb")
#         camera_bp.set_attribute("image_size_x", f"{self.img_width}")
#         camera_bp.set_attribute("image_size_y", f"{self.img_height}")
#         camera_bp.set_attribute("fov", "100")
#         camera_transform = carla.Transform(carla.Location(x=1.3, z=2.3))
#         self.camera = self.world.spawn_actor(
#             camera_bp, camera_transform, attach_to=self.ego_vehicle
#         )
#         self.camera.listen(self._process_image)

#         # Setup collision and lane invasion sensors
#         collision_bp = self.blueprint_library.find("sensor.other.collision")
#         self.collision_sensor = self.world.spawn_actor(
#             collision_bp, carla.Transform(), attach_to=self.ego_vehicle
#         )
#         self.collision_sensor.listen(self._on_collision)

#         lane_invasion_bp = self.blueprint_library.find("sensor.other.lane_invasion")
#         self.lane_invasion_sensor = self.world.spawn_actor(
#             lane_invasion_bp, carla.Transform(), attach_to=self.ego_vehicle
#         )
#         self.lane_invasion_sensor.listen(self._on_lane_invasion)

#         # # Attach LiDAR sensor to the ego vehicle
#         # lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
#         # # lidar_bp.set_attribute('range', '100')  # 100 meters range
#         # lidar_bp.set_attribute('channels', '32')  # 32 channels
#         # lidar_bp.set_attribute('rotation_frequency', '10')  # 10 Hz
#         # lidar_bp.set_attribute('upper_fov', '0')  # Upper field of view
#         # lidar_bp.set_attribute('lower_fov', '-30')  # Lower field of view
#         # lidar_bp.set_attribute('points_per_second', '100000')
#         # lidar_bp.set_attribute('dropoff_general_rate', '0.0')  # No points dropoff

#         # Tick the world
#         self.world.tick()

#         # Save all actors
#         self.actors = [
#             self.ego_vehicle,
#             self.camera,
#             self.collision_sensor,
#             self.lane_invasion_sensor,
#         ]

#         # If in debug mode, enable autopilot
#         if self.debug:
#             if self.ego_vehicle is not None:
#                 self.ego_vehicle.set_autopilot(True)

#         # # Observations are dictionaries with the sensor data
#         # self.observation_space = spaces.Dict(
#         #     {
#         #         "rbg": spaces.Box(
#         #             0, 255, shape=(self.img_height, self.img_width, 3), dtype=np.uint8
#         #         ),
#         #     }
#         # )

#         # Observations are dictionaries with the sensor data
#         self.observation_space = spaces.Box(
#             0, 255, shape=(self.img_height, self.img_width, 3), dtype=np.uint8
#         )

#         # Action corresponding to throtle, steer, brake
#         self.action_space = spaces.Box(
#             low=np.array([0, -1.0, 0]),
#             high=np.array([1.0, 1.0, 1.0]),
#             shape=(3,),
#             dtype=np.float32,
#         )

#     def _get_obs(self):
#         with sem:
#             self.img_cache = self.img_captured
#             # return {"rbg": img}
#             return self.img_cache

#     def _tick(self):
#         self.world.tick()
#         self.frame += 1
#         # print(f"Time: {self.frame*self.fixed_delta_seconds}s")

#     def _process_image(self, image):
#         with sem:
#             self.img_captured = np.array(image.raw_data, dtype=np.dtype("uint8"))
#             self.img_captured = np.reshape(
#                 self.img_captured, (self.img_height, self.img_width, 4)
#             )
#             self.img_captured = self.img_captured[:, :, :3]
#             self.img_captured = self.img_captured[:, :, ::-1]

#     def _on_collision(self, event):
#         self.collision = event

#     def _on_lane_invasion(self, event):
#         self.lane_invasion = event

#     def _follow_agent(self):
#         # Get the spectator from the world
#         spectator = self.world.get_spectator()

#         # Get the car's current transform
#         car_transform = self.ego_vehicle.get_transform()

#         # Modify the transform to move the spectator
#         car_transform.location.z = 50
#         car_transform.rotation.pitch = -70

#         # Set the spectator's transform
#         # spectator.set_transform(car_transform)
#         spectator.set_transform(self.camera.get_transform())

#     def reset(self, seed=None):
#         # Get the current time
#         now = datetime.now()
#         with open("log.txt", "a") as file:
#             file.write(f"Reset environment at {now.hour}:{now.minute}\n")
#         # We need the following line to seed self.np_random
#         super().reset(seed=seed)

#         self.ego_vehicle.set_transform(self.chosen_spawn_point)

#         # Reset the data
#         self.frame = 0
#         self.images = []
#         self.img_captured = None
#         self.collision = None
#         self.lane_invasion = None

#         # Tick the world
#         self.world.tick()

#         # Wait for image to be captured
#         while self.img_captured is None:
#             pass

#         # Get initial observation
#         observation = self._get_obs()

#         return observation, {}

#     def step(self, action):
#         print(action)
#         self._follow_agent()
#         # Set vehicle control
#         control = carla.VehicleControl(
#             throttle=float(action[0]), steer=float(action[1]), brake=float(action[2])
#         )
#         self.ego_vehicle.apply_control(control)

#         # Tick the world
#         self._tick()

#         # Calculate the reward
#         terminated = False
#         truncated = False

#         if self.collision:
#             terminated = True
#             reward = -10.0
#             self.collision = None
#             return self._get_obs(), reward, terminated, truncated, {}

#         reward = float(action[0] - abs(action[1]) - action[2])
#         if self.lane_invasion:
#             reward -= 1
#             self.lane_invasion = None

#         if self.frame >= 1000:
#             truncated = True
#             return self._get_obs(), reward, terminated, truncated, {}

#         return self._get_obs(), reward, terminated, truncated, {}

#     def destroy(self):
#         for actor in self.actors:
#             if actor is not None:
#                 if isinstance(actor, carla.Sensor):
#                     actor.stop()
#                     actor.destroy()
#                 else:
#                     actor.destroy()
#         self.actors = []

#     def close(self):
#         # # Save images and controls
#         # data = np.array(self.images)
#         # np.save("images.npy", data)
#         # # np.savez_compressed('images.npz', array=data)
#         # data = np.array(self.controls)
#         # np.save("controls.npy", data)
#         # # np.savez_compressed('controls.npz', array=data)

#         # Ensure synchronous mode is turned off
#         settings = self.world.get_settings()
#         settings.synchronous_mode = False
#         self.world.apply_settings(settings)

#         self.destroy()


if __name__ == "__main__":
    env = CarlaEnvContinuous(debug=True)
    env.reset()
    try:
        # Get the start time
        start_time = time.time()
        while time.time() - start_time < 100:
            # obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            obs, reward, terminated, truncated, _ = env.step([1.0, 1.0])
            print(reward)
            # print(obs.shape)
            if terminated or truncated:
                env.reset()
        print(f"Time elapsed: {time.time() - start_time}s")
    finally:
        env.close()
