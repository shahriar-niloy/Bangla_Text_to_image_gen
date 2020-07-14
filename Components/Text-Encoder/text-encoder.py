#
#   Text encoder: This is a bidirectional RNN (LSTM/GRU)
#   
#

import sys
import os
sys.path.append(os.getcwd())
from config import config 

import tensorflow as tf
# print('The value of text emcpder', config.TEXT_ENCODER['RNN_TYPE'])

class Encoder(tf.keras.Model):
    def __init__(self, vocabulary_size, embedding_dim, encoder_units, batch_size, rnn_type='LSTM'):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.encoder_units = encoder_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        if rnn_type == 'LSTM':
            self.recurrent = tf.keras.layers.LSTM(self.encoder_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        elif rnn_type == 'GRU':
            self.recurrent = tf.keras.layers.GRU(self.encoder_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        else:
            raise Exception('Invalid RNN Type')
        self.recurrent = tf.keras.layers.Bidirectional(self.recurrent)
    
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.recurrent(x, initial_state=hidden)
        return output, state
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.encoder_units))
