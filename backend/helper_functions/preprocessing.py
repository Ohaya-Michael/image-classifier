from tkinter import Image

import tensorflow as tf
import numpy as np
import os


def preprocess_image(img):
    """
    Preprocess the image for model input.
    Args:
        img (PIL.Image): The image to preprocess.
    Returns:
        np.array: Preprocessed image ready for model input.
    """
    # 1. Resize image to match model input shape (240, 240, 3)
    img = img.resize((240, 240))
    
    # 2. Convert image to array
    img_array = tf.keras.utils.img_to_array(img)  # Shape: (240, 240, 3)
    
    # 3. Add batch dimension (1, 240, 240, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array