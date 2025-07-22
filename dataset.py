import os
import numpy as np
from PIL import Image
import tensorflow as tf
from labels import LABELS

def preprocess_image(image_path):
    # Open, convert to grayscale, resize if needed, and normalize
    image = Image.open(image_path).convert('L')  # Grayscale
    image = image.resize((64, 64))  # Ensure size
    image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0,1]
    image_array = np.expand_dims(image_array, axis=-1)  # Shape: (64, 64, 1)
    return image_array

def get_label(filename):
    label_name = ''.join([c for c in filename if not c.isdigit()]).replace('.png', '')
    return LABELS[label_name.lower()]

def load_dataset(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    image_paths = [os.path.join(folder_path, f) for f in image_files]
    labels = [get_label(f) for f in image_files]

    images = [preprocess_image(p) for p in image_paths]
    images = np.stack(images, axis=0)
    labels = np.array(labels, dtype=np.int32)
    return images, labels

def make_tf_dataset(folder_path, batch_size=16, shuffle=True):
    images, labels = load_dataset(folder_path)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))
    dataset = dataset.batch(batch_size)
    return dataset

