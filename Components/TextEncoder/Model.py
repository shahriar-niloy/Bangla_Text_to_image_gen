#
#   Text encoder: This is a bidirectional RNN (LSTM/GRU)
#   
#

import sys
import os
sys.path.append(os.getcwd())
from config import config 

import tensorflow as tf

class TextEncoder(tf.keras.Model):
    def __init__(self, vocabulary_size, embedding_dim, encoder_units, batch_size, rnn_type='LSTM'):
        super(TextEncoder, self).__init__()
        self.batch_size = batch_size
        self.encoder_units = encoder_units
        self.embedding = tf.keras.layers.Embedding(vocabulary_size, embedding_dim)
        if rnn_type == 'LSTM':
            self.recurrent = tf.keras.layers.LSTM(self.encoder_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        elif rnn_type == 'GRU':
            self.recurrent = tf.keras.layers.GRU(self.encoder_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        else:
            raise Exception('Invalid RNN Type')
        self.recurrent = tf.keras.layers.Bidirectional(self.recurrent)
    
    def call(self, x):
        x = self.embedding(x)
        entire_hidden_state, forward_hidden, forward_cell, backward_hidden, backward_cell = self.recurrent(x)

        word_vector = entire_hidden_state       # Dimension [Number of captions, max number of words in captions, 2 * encoder units] multiply with two because biderectional => forward and backward
        sent_vector = tf.concat([forward_hidden, backward_hidden], axis=-1)            # Dimension [number of captions, encoder units]
        
        return word_vector, sent_vector
