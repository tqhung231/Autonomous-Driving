import random

import carla
import cv2
import numpy as np


class CarlaEnvironment:
    def __init__(self, host="127.0.0.1", port=2000, debug=False):
        self.client = carla.Client(host, port)
        self.client.set_timeout(30.0)
        self.client.load_world("Town06")

        self.world = self.client.get_world()

        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / 60.0  # FPS
        settings.no_rendering_mode = True
        self.world.apply_settings(settings)

        self.debug = debug
        self.blueprint_library = self.world.get_blueprint_library()
        self.ego_vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None

        # Set image size
        self.im_width = 640
        self.im_height = 480

        self.image_captured = False

        self.actors = []

    def reset(self):
        self.destroy()

        spawn_points = self.world.get_map().get_spawn_points()
        chosen_spawn_point = random.choice(spawn_points)

        # Setup ego vehicle
        vehicle_bp = self.blueprint_library.find("vehicle.tesla.model3")
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, chosen_spawn_point)

        # Setup RGB camera
        camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", f"{self.im_width}")
        camera_bp.set_attribute("image_size_y", f"{self.im_height}")
        camera_bp.set_attribute("fov", "100")
        # camera_bp.set_attribute("sensor_tick", str(1.0 / 60.0))
        camera_transform = carla.Transform(carla.Location(x=1.3, z=2.3))
        self.camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.ego_vehicle
        )
        self.camera.listen(lambda image: self.process_image(image))

        # Setup collision and lane invasion sensors
        collision_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.ego_vehicle
        )
        self.collision_sensor.listen(lambda event: self.on_collision(event))

        lane_invasion_bp = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_invasion_sensor = self.world.spawn_actor(
            lane_invasion_bp, carla.Transform(), attach_to=self.ego_vehicle
        )
        self.lane_invasion_sensor.listen(lambda event: self.on_lane_invasion(event))

        self.actors = [
            self.ego_vehicle,
            self.camera,
            self.collision_sensor,
            self.lane_invasion_sensor,
        ]

        # Tick the world
        self.world.tick()

        # If in debug mode, disable synchronous mode and enable autopilot
        if self.debug:
            if self.ego_vehicle is not None:
                self.ego_vehicle.set_autopilot(True)

    def destroy(self):
        for actor in self.actors:
            if actor is not None:
                actor.destroy()
        self.actors = []

    def process_image(self, image):
        # Set the image captured flag
        self.image_captured = True

        # Convert the image to a numpy array
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (self.im_height, self.im_width, 4))

        # Drop the alpha channel and convert from BGR to RGB
        array = array[:, :, :3]
        # array = array[:, :, ::-1]  # cv2 uses BGR

        # Display the image using cv2
        cv2.imshow("CARLA RGB Camera", array)
        cv2.waitKey(1)  # Introduce a 1ms delay
        print("Image captured!")

    def on_collision(self, event):
        print("Collision detected!")

    def on_lane_invasion(self, event):
        print("Lane invasion detected!")

    def step(self, action):
        self.image_captured = False
        self.follow_agent()
        self.world.tick()

    def follow_agent(self):
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

    def close(self):
        # Ensure synchronous mode is turned off
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

        self.destroy()


if __name__ == "__main__":
    env = CarlaEnvironment(debug=True)
    env.reset()
    try:
        while True:
            if env.image_captured:
                env.step(None)
    finally:
        env.close()
