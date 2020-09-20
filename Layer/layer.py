import tensorflow as tf

class Dense(tf.keras.layers.Layer):
    def __init__(self, units):
        self.units = units
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units, name=name)

    def call(self, x):
        x = self.flatten(x)
        x = self.dense(x)
        return x

