import tensorflow as tf
import sys
import os
sys.path.append(os.getcwd())
from config import config 

# Input: (Word Embedding, Hidden State)
# Algorithm: 
    # Transform word embedding into a shape that makes it easier to combine with another matrix
    # For each region in the hidden state, Calculate Cj
        # Cj = summation[i=0 -> T]( B[j,i] * e' )
        # where B[j,i] = exp(s'[j,i]) / ( Summation[k=0 -> T] (exp(s'[j,k]) )) is the softmax probability of ith word being responsible for paiting jth subregion in the image 

class Attention(tf.keras.Model):
    def __init__(self, common_dimension):
        super(Attention, self).__init__()
        self.common_dimension = common_dimension
        self.conv1x1 = tf.keras.layers.Conv2D(filters=self.common_dimension, kernel_size=1, strides=1)
        pass

    def call(self, hidden, word_vectors):
        '''
            word_vectors: Dimension = [batch, seq_len, word_feature]
        '''
        batch, height, width, channel = hidden.get_shape()
        subregions = height * width
        _, seq_len, _ = word_vectors.get_shape()

        # hidden = [batch, height, weight, channel] -> [batch, sub_regions, common_dimension] where sub_regions = height * weight and channel = common_dimension
        hidden = tf.reshape(hidden, shape=[batch, subregions, self.common_dimension])
        # word_vectors = [batch, sequence length, rnn unit * number of direction] == Dimension expansion operation => [batch, 1, sequence length, rnn unit * number of direction]
        word_vectors = tf.expand_dims(word_vectors, axis=1)
        # word_vectors = [batch, 1, sequence length, rnn unit * number of direction] == conv 1x1 operation => [batch, 1, sequence length, common_dimension]
        word_vectors = self.conv1x1(word_vectors)
        # word_vectors = [batch, 1, sequence length, common_dimension] == Dimension expansion operation => [batch, sequence length, common_dimension]
        word_vectors = tf.squeeze(word_vectors, axis=1)
        # word_vectors = [batch, sequence length, common_dimension] == transpose operation => [batch, common_dimension, sequence length]
        word_vectors = tf.transpose(word_vectors, perm=[0, 2, 1])   

        # Construct hidden x word_vector matrix 
        # [batch, sub_regions, common_dimension] x [batch, common_dimension, sequence length] = [batch, sub_regions, seq_len]
        hidden_word_matrix = tf.matmul(hidden, word_vectors)
        hidden_word_matrix = tf.reshape(hidden_word_matrix, shape=[batch * subregions, seq_len])
        hidden_word_matrix = tf.nn.softmax(hidden_word_matrix)
        hidden_word_matrix = tf.reshape(hidden_word_matrix, shape=[batch, subregions, seq_len])
        hidden_word_matrix = tf.transpose(hidden_word_matrix, perm=[0, 2, 1])   

        # Construct Final word context vector
        # [batch, common_dimension, sequence length] x [batch, seq_length, sub_regions] = [batch, common_dimension, sub_regions]
        word_context = tf.matmul(word_vectors, hidden_word_matrix)

        return word_context
