import glob
import os

import cv2
import numpy as np
from natsort import os_sorted


def images_to_video(image_folder, video_name, fps):
    # Get all files from the folder
    img_array = []
    for filename in os_sorted(glob.glob(os.path.join(image_folder, "*.png"))):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    # Create a video writer object
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"XVID"), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def data_to_video(data_path, video_name, frame_shape, fps):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(video_name, fourcc, fps, frame_shape)

    # Load numpy array
    image_array = np.load(data_path, allow_pickle=True)

    # Iterate over each 1D image and write it to the video
    for image in image_array:
        image = np.reshape(image, (frame_shape[1], frame_shape[0], 4))
        image = image[:, :, :3]
        out.write(image)

    # Release everything if job is finished
    out.release()


# images_to_video("test", "vid.avi", 20)
data_to_video("images.npy", "vid.avi", (640, 480), 20)
