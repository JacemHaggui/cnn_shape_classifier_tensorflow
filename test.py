import os
import numpy as np
from PIL import Image
import tensorflow as tf
from model import CNN
from labels import LABELS

# Reverse the label dictionary for decoding
INT_TO_LABEL = {v: k for k, v in LABELS.items()}

# Path to your test images
test_folder = "train"

# Load the trained model, providing the custom class in custom_objects
model = tf.keras.models.load_model("Models/CNN_model.keras")

correct = 0
total = 0

def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((64, 64))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=-1)  # (64, 64, 1)
    image_array = np.expand_dims(image_array, axis=0)   # (1, 64, 64, 1)
    return image_array

for filename in sorted(os.listdir(test_folder)):
    if filename.endswith(".png"):
        image_path = os.path.join(test_folder, filename)
        image_array = preprocess_image(image_path)

        output = model.predict(image_array)
        predicted_class = np.argmax(output, axis=1)[0]
        predicted_label = INT_TO_LABEL[predicted_class]

        expected_label = ''.join([c for c in filename if not c.isdigit()]).replace('.png', '')

        print(f"{filename} → Predicted: {predicted_label}, Expected: {expected_label}")

        if predicted_label == expected_label:
            correct += 1
        total += 1

print(f"\nCorrect: {correct}/{total} ({100 * correct / total:.2f}%)")