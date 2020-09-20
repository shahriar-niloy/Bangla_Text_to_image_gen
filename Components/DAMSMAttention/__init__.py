import sys
import os
import tensorflow as tf 

sys.path.append(os.getcwd())

from config import config 


class DAMSMAttention(tf.keras.layers.Layer): 
    def __init__(self):
        super(DAMSMAttention, self).__init__()
        self.gamma1 = config.DAMSM['GAMMA1']

    def call(self, image_feature, word_vector): 
        '''
            image_feature: 
                Dimension:  batch x 17 x 17 x multimodal dimension 
            word_vector:    
                Dimension: batch x number of words x direction * rnn unit
        '''

        batch_size = word_vector.shape[0]
        seq_len = word_vector.shape[1]
        height, width = image_feature.shape[1], image_feature.shape[2]
        
        # Similarity Matrix: Matrix Multiply image feature and word vector 
        context = tf.reshape(image_feature, [batch_size, height * width, -1])        # Getting rid of the channel dimension so that we can apply batch matrix multiplication 
        attn = tf.matmul(context, word_vector, transpose_b=True)                      # Building Similarity matrix. Dimension: [batch_size x (number of subregions = 289) x number of words]

        # Softmanx Similarity: Apply softmax on the last dimension, the word number dimension
        attn = tf.reshape(attn, [batch_size * height * width, -1])
        attn = tf.nn.softmax(attn)      # default axis=-1                                   # Building Softmax Similarity matrix. Dimension: [batch_size x (number of subregions = 289) x number of words]
        attn = tf.reshape(attn, [batch_size, height * width, -1])

        # Alpha: Apply Softmax on the sub region dimension 
        attn = tf.transpose(attn, perm=[0, 2, 1])
        attn = tf.reshape(attn, [batch_size * seq_len, height * width])
        attn = self.gamma1 * attn                                                           # Multiply softmax similarity with gamma1 constant  Dimension: [batch_size x number of words x (number of subregions = 289)]
        attn = tf.nn.softmax(attn)
        attn = tf.reshape(attn, [batch_size, seq_len, height * width])
        
        # Region context vector: Matrix multiplication image feature and attn 
        # [bs, hw, ndf] x [bs, seq_len, hw]
        # [bs, ndf, hw] x [bs, hw, seq_len]
        region_context = tf.matmul(context, attn, transpose_a=True, transpose_b=True)        # Dimension [bs, ndf, seq_len]

        return region_context