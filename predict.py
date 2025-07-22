import numpy as np
from PIL import Image
import tensorflow as tf
from model import CNN
from labels import LABELS

# Reverse the label dictionary for decoding
INT_TO_LABEL = {v: k for k, v in LABELS.items()}

# Path to your single test image
image_path = "hand-drawn.png"

# Load the trained model
model = tf.keras.models.load_model("CNN_model.keras", custom_objects={'CNN': CNN})

def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((64, 64))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=-1)  # (64, 64, 1)
    image_array = np.expand_dims(image_array, axis=0)   # (1, 64, 64, 1)
    return image_array

image_array = preprocess_image(image_path)

output = model.predict(image_array)
predicted_class = np.argmax(output, axis=1)[0]
predicted_label = INT_TO_LABEL[predicted_class]

print(f"Image: {image_path} → Predicted label: {predicted_label}")