import tensorflow as tf 

class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, momentum=0.9, epsilon=1e-5, name="BatchNorm"):
        super(BatchNorm, self).__init__()
        self.batchNorm = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon, name=name)

    def call(self, x):
        x = self.batchNorm(x)
        return x
