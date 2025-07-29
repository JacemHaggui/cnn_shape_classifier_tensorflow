import os
import numpy as np
from PIL import Image
import tensorflow as tf
from labels import LABELS  # Your label dict

INT_TO_LABEL = {v: k for k, v in LABELS.items()}

test_folder = "train"
tflite_model_path = "Models/CNN_model_quantised_post_training.tflite"

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((64, 64))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=-1)  # (64, 64, 1)
    image_array = np.expand_dims(image_array, axis=0)   # (1, 64, 64, 1)
    return image_array

correct = 0
total = 0

for filename in sorted(os.listdir(test_folder)):
    if filename.endswith(".png"):
        image_path = os.path.join(test_folder, filename)
        image_array = preprocess_image(image_path)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], image_array)

        # Run inference
        interpreter.invoke()

        # Get output tensor
        output = interpreter.get_tensor(output_details[0]['index'])  # shape (1, 5)
        predicted_class = np.argmax(output, axis=1)[0]
        predicted_label = INT_TO_LABEL[predicted_class]

        # Extract expected label from filename (strip digits and .png)
        expected_label = ''.join([c for c in filename if not c.isdigit()]).replace('.png', '')

        print(f"{filename} → Predicted: {predicted_label}, Expected: {expected_label}")

        if predicted_label == expected_label:
            correct += 1
        total += 1

print(f"\nCorrect: {correct}/{total} ({100 * correct / total:.2f}%)")