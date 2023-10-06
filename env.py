import os
import random
import time

import carla
import cv2
import numpy as np


DELTA_SECONDS = 0.05


class CarlaEnvironment:
    def __init__(self, host="127.0.0.1", port=2000, debug=False):
        self.client = carla.Client(host, port)
        self.client.set_timeout(30.0)
        self.client.load_world("Town04")

        self.world = self.client.get_world()

        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DELTA_SECONDS  # FPS
        # settings.no_rendering_mode = True
        self.world.apply_settings(settings)
        self.fixed_delta_seconds = DELTA_SECONDS

        self.debug = debug
        self.blueprint_library = self.world.get_blueprint_library()
        self.ego_vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None

        # Set image size
        self.img_width = 640
        self.img_height = 480

        self.frame = 0
        self.img_captured = (0, None)
        self.actors = []
        self.images = [(0, None)]

    def tick(self):
        self.world.tick()
        self.frame += 1
        # print(f"Time: {self.frame*self.fixed_delta_seconds}s")

    def reset(self):
        self.destroy()

        self.frame = 0
        self.img_captured = (0, None)

        spawn_points = self.world.get_map().get_spawn_points()
        chosen_spawn_point = random.choice(spawn_points[345:353])

        # Setup ego vehicle
        vehicle_bp = self.blueprint_library.find("vehicle.tesla.model3")
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, chosen_spawn_point)

        # Setup RGB camera
        camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", f"{self.img_width}")
        camera_bp.set_attribute("image_size_y", f"{self.img_height}")
        camera_bp.set_attribute("fov", "100")
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
        self.tick()

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
        self.images.append(
            (self.frame, np.frombuffer(image.raw_data, dtype=np.dtype("uint8")))
        )

    # def process_image(self, image):
    #     # Convert the image to a numpy array
    #     array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    #     array = np.reshape(array, (self.img_height, self.img_width, 4))

    #     # Drop the alpha channel and convert from BGR to RGB
    #     array = array[:, :, :3]
    #     # array = array[:, :, ::-1]  # cv2 uses BGR

    #     # # Display the image using cv2
    #     # cv2.imshow("CARLA RGB Camera", array)
    #     # cv2.waitKey(1)  # Introduce a 1ms delay
    #     # # print("Image captured!")

    #     # Save the image
    #     self.img_captured = (self.frame, array)

    #     # Save image to file only if the image is not exist
    #     if not os.path.exists(f"test/{self.frame}.png"):
    #         cv2.imwrite(f"test/{self.frame}.png", array)
    #         self.frame += 1
    #     else:
    #         print("Image already exists!")

    def on_collision(self, event):
        print("Collision detected!")

    def on_lane_invasion(self, event):
        print("Lane invasion detected!")

    def step(self, action):
        # while self.img_captured[0] != self.frame:
        #     continue
        while self.images[-1][0] != self.frame:
            continue
        self.follow_agent()
        self.tick()

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
        print(len(self.images))
        print(f"{len(self.images) * DELTA_SECONDS}s")
        # data = np.array(self.images)
        # np.save("data.npy", data)
        # np.savez_compressed('array.npz', array=data)
        # Ensure synchronous mode is turned off
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

        self.destroy()


if __name__ == "__main__":
    env = CarlaEnvironment(debug=True)
    env.reset()
    # Get the start time
    start_time = time.time()
    try:
        while time.time() - start_time < 300:
            env.step(None)
    finally:
        env.close()
