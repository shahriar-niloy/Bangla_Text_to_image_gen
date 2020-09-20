import sys
import os
sys.path.append(os.getcwd())
from config import config 
import tensorflow as tf 

# Output Dimension: [Number of dimensions, conditional dimension]
class ConditionalAugmentation(tf.keras.Model):
    def __init__(self, conditional_dim, name="ConditionalAugmentation"):
        super(ConditionalAugmentation, self).__init__()
        self.conditional_dim = conditional_dim
        self.dense = tf.keras.layers.Dense(self.conditional_dim * 2)
    
    def call(self, sentence_vector):
        x = self.dense(sentence_vector)
        mean = x[:, :self.conditional_dim]
        std = x[:, self.conditional_dim:]
        eps = tf.random.normal(tf.shape(mean), mean=0.0, stddev=1.0, dtype=tf.float32)
        condition_code = mean + tf.exp(std * 0.5) * eps
        return condition_code, mean, std                                   

    def kl_loss(self, mean, std):
        # shape : [batch_size, channel]
        loss = 0.5 * tf.reduce_sum(tf.square(mean) + tf.exp(std) - 1 - std, axis=-1)
        loss = tf.reduce_mean(loss)
        return loss