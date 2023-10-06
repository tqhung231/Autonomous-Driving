import glob
import os

import cv2
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


images_to_video("test", "vid.avi", 20)
