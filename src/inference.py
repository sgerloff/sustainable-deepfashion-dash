import tensorflow as tf
from PIL import Image
import numpy as np

import base64, io

def convert_base64_string_to_array(base64_string):
    decoded = base64.b64decode(base64_string.split(",")[1])
    bytes_image = io.BytesIO(decoded)
    image = Image.open(bytes_image, formats=None).convert('RGB')
    return np.array(image)


def distance(vec1, vec2, metric="L2"):
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    vec1 = normalize_vector(vec1)
    vec2 = normalize_vector(vec2)

    if metric == "L2":
        return np.linalg.norm((vec1 - vec2))
    if metric == "angular":
        return np.maximum(1. - np.dot(vec1, vec2), 0.0)


def normalize_vector(vec):
    return vec / np.linalg.norm(vec)


class ModelInference:
    def __init__(self, model_path):
        self.loaded_model = tf.saved_model.load(model_path)
        self.embedding_extractor = self.loaded_model.signatures["serving_default"]
        self.preprocessor, self.input_shape = self.get_model_specifics()

    def get_model_specifics(self):
        return (lambda x: (x/127.5)-1.), (224,224,3)

    def predict(self, base64_string):
        array = convert_base64_string_to_array(base64_string)

        image = tf.expand_dims(array, 0)
        image = tf.image.resize(image, [self.input_shape[0], self.input_shape[1]])
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = self.preprocessor(image)

        pred = self.embedding_extractor(image)
        return pred["sequential_1"].numpy()

    def get_metric(self):
        return "angular"


