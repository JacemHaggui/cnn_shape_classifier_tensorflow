import tensorflow as tf
from tensorflow.keras import layers, models

class CNN(tf.keras.Model):
    def __init__(self, **kwargs):   # Accept arbitrary kwargs
        super(CNN, self).__init__(**kwargs)  # Pass them to base class

        # Convolutional layer 1: input=1 channel, output=8 filters
        self.conv1 = layers.Conv2D(filters=8, kernel_size=3, padding='same', activation='relu')
        self.pool1 = layers.MaxPooling2D(pool_size=2, strides=2)

        # Convolutional layer 2: input=8, output=16 filters
        self.conv2 = layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')
        self.pool2 = layers.MaxPooling2D(pool_size=2, strides=2)

        # Flatten + Dense layers
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(5)  # No activation if you're using from_logits=True in loss

    def call(self, x):
        x = self.conv1(x)  # [batch, 64, 64, 8]
        x = self.pool1(x)  # [batch, 32, 32, 8]
        x = self.conv2(x)  # [batch, 32, 32, 16]
        x = self.pool2(x)  # [batch, 16, 16, 16]
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)  # logits