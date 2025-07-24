import tensorflow as tf
from model import CNN  # custom class definition must be available

# Load the trained model and register the custom class
model = tf.keras.models.load_model("CNN_model.keras", custom_objects={"CNN": CNN})

# Force the model to build by calling it once
_ = model(tf.zeros((1, 64, 64, 1)))  # Dummy input with correct shape

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Uncomment the next line to enable quantization (this will be post-training quantization)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enables quantization
tflite_model = converter.convert()

# Save .tflite model
with open("CNN_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved.")