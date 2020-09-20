import sys
import os
import tensorflow as tf 

sys.path.append(os.getcwd())

from config import config 


class ImageEncoder(tf.keras.Model):
    def __init__(self, embed_dim): 
        super(ImageEncoder, self).__init__()
        self.embed_dim = embed_dim

        self.inception_v3_preprocess = tf.keras.applications.inception_v3.preprocess_input
        self.inception_v3 = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        self.inception_v3.trainable = False 

        self.inception_v3_mixed7 = tf.keras.Model(inputs=self.inception_v3.input, outputs=self.inception_v3.get_layer('mixed7').output)
        self.inception_v3_mixed7.trainable = False

        self.emb_feature = tf.keras.layers.Conv2D(filters=self.embed_dim, kernel_size=1, strides=1, use_bias=True)
        self.emb_code = tf.keras.layers.Dense(self.embed_dim)

    def call(self, x, training=True): 
        x = ((x + 1) / 2) * 255.0
        x = tf.image.resize(x, size=[299, 299], method=tf.image.ResizeMethod.BILINEAR)
        x = self.inception_v3_preprocess(x)

        # get sentence vector
        code = self.inception_v3(x)
        # Get sub region vectors
        feature = self.inception_v3_mixed7(x)

        # Change the dimension of sentence code and sub region vector 
        code = self.emb_code(code)
        feature = self.emb_feature(feature)

        return feature, code 

