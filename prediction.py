# from json import load
from PIL import Image
from io import BytesIO

import numpy as np
import tensorflow as tf

INPUT_SHAPE = (256,256)
MODEL = tf.keras.models.load_model('models/sample-thesis-dataset-v3_simple_BN-DO.keras')

def check_version():
    try:
        return tf.__version__
    except Exception as e:
        return "library not found"

def read_image(image_encoded):
    pil_image = Image.open(BytesIO(image_encoded)).convert('RGB')
    return pil_image

def preprocess_image(image: Image.Image):
    resize = tf.image.resize(image, INPUT_SHAPE)
    ready_image = np.expand_dims(resize/255, 0)
    return ready_image

def predict_image(image: np.ndarray):
    y = MODEL.predict(image)
    y = np.argmax(y)
    img_class = "Road" if y == 3 else "Not Road"
    return img_class

