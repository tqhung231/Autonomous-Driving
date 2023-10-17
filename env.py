import threading
import time

import carla
import gymnasium
import numpy as np
import pygame
from gymnasium import spaces

event = threading.Event()

DELTA_SECONDS = 0.05


class CarlaEnv(gymnasium.Env):
    metadata = {"render_modes": ["true", "false"], "render_fps": 20}

    def __init__(self, debug=False, render_mode=True):
        # Connect to the server
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(30.0)
        self.client.load_world("Town04")

        # Get the world
        self.world = self.client.get_world()

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
        self.img_captured = None
        self.actors = []
        self.collision = None
        self.lane_invasion = None

        # Set actors
        self.blueprint_library = self.world.get_blueprint_library()

        # Set ego vehicle spawn point
        spawn_points = self.world.get_map().get_spawn_points()
        # chosen_spawn_point = random.choice(spawn_points[345:353])
        self.chosen_spawn_point = spawn_points[347]

        # Setup ego vehicle
        vehicle_bp = self.blueprint_library.find("vehicle.tesla.model3")
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, self.chosen_spawn_point)

        # Setup RGB camera
        camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", f"{self.img_width}")
        camera_bp.set_attribute("image_size_y", f"{self.img_height}")
        camera_bp.set_attribute("fov", "100")
        camera_transform = carla.Transform(carla.Location(x=1.3, z=2.3))
        self.camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.ego_vehicle
        )
        self.camera.listen(self._process_image)

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

        # Tick the world
        self.world.tick()

        # Save all actors
        self.actors = [
            self.ego_vehicle,
            self.camera,
            self.collision_sensor,
            self.lane_invasion_sensor,
        ]

        # If in debug mode, enable autopilot
        if self.debug:
            if self.ego_vehicle is not None:
                self.ego_vehicle.set_autopilot(True)

        # Observations are dictionaries with the sensor data
        self.observation_space = spaces.Dict(
            {
                "rbg": spaces.Box(
                    0, 255, shape=(self.img_height, self.img_width, 3), dtype=np.uint8
                ),
            }
        )

        # Action corresponding to throtle, steer, brake
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def _get_obs(self):
        event.wait()  # Wait for image retrieval
        img = self.img_captured
        event.clear()  # Clear the event
        self.img_captured = None  # Reset the image
        return {"rbg": img}

    def _tick(self):
        self.world.tick()
        self.frame += 1
        print(f"Time: {self.frame*self.fixed_delta_seconds}s")

    def _process_image(self, image):
        if self.img_captured is None:
            self.img_captured = np.array(image.raw_data, dtype=np.dtype("uint8"))
            self.img_captured = np.reshape(
                self.img_captured, (self.img_height, self.img_width, 4)
            )
            self.img_captured = self.img_captured[:, :, :3]
            self.img_captured = self.img_captured[:, :, ::-1]
            event.set()

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
        # spectator.set_transform(car_transform)
        spectator.set_transform(self.camera.get_transform())

    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.ego_vehicle.set_transform(self.chosen_spawn_point)

        # Reset the data
        self.frame = 0
        self.img_captured = None
        self.images = []
        self.controls = []
        self.collision = None
        self.lane_invasion = None

        # Tick the world
        self.world.tick()

        # Get initial observation
        observation = self._get_obs()

        return observation, {}

    def step(self, action):
        self._follow_agent()

        # Set vehicle control
        if action[0] < 0:
            action[0] = 0
        if action[2] < 0:
            action[2] = 0
        control = carla.VehicleControl(
            throttle=float(action[0]), steer=float(action[1]), brake=float(action[2])
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
        if self.lane_invasion:
            reward -= 1
            self.lane_invasion = None

        if self.frame >= 1000:
            truncated = True
            return self._get_obs(), reward, terminated, truncated, {}

        return self._get_obs(), reward, terminated, truncated, {}

    def destroy(self):
        for actor in self.actors:
            if actor is not None:
                if isinstance(actor, carla.Sensor):
                    actor.stop()
                    actor.destroy()
                else:
                    actor.destroy()
        self.actors = []

    def close(self):
        # # Save images and controls
        # data = np.array(self.images)
        # np.save("images.npy", data)
        # # np.savez_compressed('images.npz', array=data)
        # data = np.array(self.controls)
        # np.save("controls.npy", data)
        # # np.savez_compressed('controls.npz', array=data)

        # Ensure synchronous mode is turned off
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

        self.destroy()


if __name__ == "__main__":
    env = CarlaEnv(debug=True)
    env.reset()
    try:
        # Get the start time
        start_time = time.time()
        while time.time() - start_time < 1:
            env.step([0, 0, 0])
        print(f"Time elapsed: {time.time() - start_time}s")
    finally:
        env.close()
