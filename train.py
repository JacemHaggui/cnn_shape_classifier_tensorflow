import os
import tensorflow as tf
from dataset import make_tf_dataset
from model import CNN

# Load dataset
train_folder = "train"  # Update with your actual train folder path
dataset = make_tf_dataset(train_folder, batch_size=32, shuffle=True)

# Create model
model = CNN()

# Compile model (loss and optimizer)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Remove existing model file if it exists
model_path = "CNN_model.keras"
if os.path.exists(model_path):
    os.remove(model_path)
    print(f"Existing {model_path} file removed.")

# Train the model
epochs = 300
model.fit(dataset, epochs=epochs)

# Save the trained model
model.save(model_path)
print(f"Model saved as {model_path}")