# Image Preprocessing

# Importing the libraries
from skimage import transform
import numpy as np

# Preprocessing the Images
def preprocess_image(frame):
    # Cropping the game window, crop maps onto [Up: Down, Left: right]
    cropped_frame = frame[15:-5, 20:-20]
    normalized_frame = cropped_frame / 255.0
    
    #Resize the frame so it fits the model
    preprocessed_frame = transform.resize(normalized_frame, [80, 80])
    preprocessed_frame = np.reshape(preprocessed_frame, [1, 80, 80])
    return preprocessed_frame