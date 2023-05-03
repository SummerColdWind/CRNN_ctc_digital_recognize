import os
import cv2

from .get_config import get_config

def get_image_shape(config_path):
    global_config, train_config, test_config = get_config(config_path)
    image_path = [image for image in os.listdir(train_config['data_dir']) if image.endswith('.png')][0]
    image = cv2.imread(os.path.join(train_config['data_dir'], image_path))
    return image.shape[:2]

