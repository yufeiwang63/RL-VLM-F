from PIL import Image
import numpy as np
import cv2

image_path = "/home/yufei/miniconda3/envs/vlm-reward/lib/python3.9/site-packages/metaworld/envs/assets_v2/textures/navy_blue.png"
image = Image.open(image_path)
image_np = np.array(image)


# Define the dimensions of the image
width, height = image_np.shape[1], image_np.shape[0]

# Create an array of the specified size with a pure green color (RGB: 0,255,0)
green_color = (0, 240, 0)
green_image = np.full((height, width, 3), green_color, dtype=np.uint8)

# Convert the array to an image
image = Image.fromarray(green_image)

# Save the image
image_path = '/home/yufei/miniconda3/envs/vlm-reward/lib/python3.9/site-packages/metaworld/envs/assets_v2/textures/green.png'
image.save(image_path)

image_path
