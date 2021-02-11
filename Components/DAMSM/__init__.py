import sys
import os
import tensorflow as tf 
import numpy as np 
from PIL import Image 

sys.path.append(os.getcwd())

from config import config 
from Components.Attention.Model import Attention
from Components.ConditionalAugmentation import ConditionalAugmentation
from Components.TextEncoder.Model import TextEncoder
from Components.ImageEncoder import ImageEncoder

def cosine_similarity(x, y):                    # --> batch_size*words_num x nef
    xy = tf.reduce_sum(x * y, axis=-1)          # Sum reduce along dimension  1 or row dimension 
    x = tf.norm(x, axis=-1)                     # Summation of x1^2 and then square root the summation. Do this along the row dimension as specified by axis = -1
    y = tf.norm(y, axis=-1)                     # This is getting the magnitude 

    similarity = (xy / ((x * y) + 1e-8))        # Apply the cosine similarity formula 

    return similarity                           # A vector of size batch_size * number_of_words 

class DAMSM(tf.keras.Model):
    def __init__(self, datasetLoader, name="DAMSM"):
        super(DAMSM, self).__init__()
        self.datasetLoader = datasetLoader
        self.dictionary_size = config.TEXT_ENCODER['DICTIONARY_SIZE']
        self.embedding_dim = config.TEXT_ENCODER['EMBEDDING_DIMENSION']
        self.encoder_units = config.TEXT_ENCODER['ENCODER_UNITS']
        self.batch_size = config.DATASET['BATCH_SIZE']
        self.checkpoint_path = config.DAMSM['CHECKPOINT_PATH']
        self.max_numer_of_checkpoints = config.DAMSM['MAX_NUMBER_OF_CHECKPOINTS']
        self.batch_size = config.DATASET['BATCH_SIZE']
        self.gamma1 = config.DAMSM['GAMMA1']
        self.learning_rate = config.DAMSM['LEARNING_RATE']
        self.save_interval = config.DAMSM['SAVE_INTERVAL']
        self.gamma2 = 5.0
        self.gamma3  = config.DAMSM['SENTENCE_LOSS_CONSTANT']
        self.smooth_lambda = config.SMOOTH['LAMBDA']
        self.build_model()

    def build_model(self):
        self.imageEncoder = ImageEncoder(self.embedding_dim)
        self.textEncoder = TextEncoder(self.dictionary_size, self.embedding_dim, self.encoder_units, self.batch_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5, beta_2=0.999, epsilon=1e-08)

    def load_model_parameters(self):
        self.checkpoint = tf.train.Checkpoint(text_encoder=self.textEncoder, image_encoder=self.imageEncoder)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_path, max_to_keep=self.max_numer_of_checkpoints)

        if self.checkpoint_manager.latest_checkpoint is not None:
            print("Loading from checkpoint....")
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

    def call(self, epochs):
        self.datasetLoader.load() 
        dataset_size = len(self.datasetLoader.get_captions())
        number_of_step = dataset_size // self.batch_size

        self.load_model_parameters()

        for current_epoch in range(epochs):
            dataset_iter = iter(self.datasetLoader)

            average_last_n_step_error = 0 
            average_epoch_error = 0

            for current_step in range(number_of_step):
                batch_captions, batch_images, class_ids = next(dataset_iter)
                real_image0, real_image1, real_image2 = batch_images 

                with tf.GradientTape() as damsm_tape:
                    word_vector, sentence_vector = self.textEncoder(batch_captions)
                    sub_region_vector, global_image_vector = self.imageEncoder(real_image2)
                    
                    word_loss = self.word_loss(sub_region_vector, word_vector, class_ids)
                    sentence_loss = self.sentence_loss(global_image_vector, sentence_vector, class_ids)

                    total_loss = word_loss + sentence_loss
                
                    average_last_n_step_error = average_last_n_step_error + total_loss
                    average_epoch_error = average_epoch_error + total_loss

                    print('Current Epoch: {}, current step: {}, Total loss: {}'.format(current_epoch, current_step, total_loss))

                damsm_train_variable = self.textEncoder.trainable_variables + self.imageEncoder.trainable_variables
                damsm_gradient = damsm_tape.gradient(total_loss, damsm_train_variable)

                self.optimizer.apply_gradients(zip(damsm_gradient, damsm_train_variable))

                if(current_step % self.save_interval == 0):
                    print("Average error of last ", self.save_interval, " steps: ", average_last_n_step_error/self.save_interval)
                    average_last_n_step_error = 0
                    self.checkpoint_manager.save()
            
            print('Epoch ', current_epoch, " error: ", average_epoch_error/number_of_step)

            self.checkpoint_manager.save()
                
    def word_loss(self, image_feature, word_vector, class_ids):
        # Build similarity matrix for all possible pairs of words in the sentence and sub regions s = e^T * v 
        # Normalize the similarity matrix using softmax function 
        # Build a attention model to compute a region context vector for each word Ci
        # Build a cosine similarity matrix with for each region context vector and regions 
        batch_size = word_vector.shape[0]
        seq_len = word_vector.shape[1]

        label = tf.cast(range(batch_size), tf.int32)
        masks = []
        similarities = []

        for i in range(batch_size): 
            mask = (class_ids.numpy() == class_ids[i].numpy()).astype(np.uint8)              # Mask samples of the same class as the current class 
            mask[i] = 0                                                                     # Unmask the current sample 
            masks.append(np.reshape(mask, newshape=[1, -1]))

            word = word_vector[i, :, :]                                     # Select the current sample (caption). This causes the batch dimension to be removed 
            word = tf.expand_dims(word, axis=0)                             # So add the batch dimension. Now the batch dimension is 1
            word = tf.tile(word, multiples=[batch_size, 1, 1])              # Now increase the batch dimension 
            
            # print('Image Feature: ', image_feature.shape)

            # Get region context vectors using DAMSM Attention. 
            region_context = self.attention(image_feature, word)            # Dimension [bs, ndf, seq_len]
            # print('Region context vector: ', region_context.shape)

            # Calculate cosine similarity 
            region_context = tf.transpose(region_context, perm=[0, 2, 1])
            word = tf.reshape(word, shape=[batch_size * seq_len, -1])                       # Convert to batch of vectors form so that we can do cosine similarity 
            region_context = tf.reshape(region_context, shape=[batch_size * seq_len, -1])   # Convert to batch of vectors form so that we can do cosine similarity 
            row_sim = cosine_similarity(word, region_context)
            row_sim = tf.reshape(row_sim, shape=[batch_size, seq_len])                      # Similarity scores for the words of one caption against all images in the batch.
                                                                                            # Like for first row = scores of words of the caption against first image 

            row_sim = tf.exp(row_sim * self.gamma2)
            row_sim = tf.reduce_sum(row_sim, axis=-1, keepdims=True)                        # Dimension: [batch_size x 1] => score for the current caption against all batch_size images 
            row_sim = tf.math.log(row_sim)

            similarities.append(row_sim)
        
        similarities = tf.concat(similarities, axis=-1)                                     # Concat on the inner most dimension. [batch_size x batch_size]. First column: First caption against all image. Second column: Second column against all images. 
        masks = tf.cast(tf.concat(masks, axis=0), tf.float32)                               # So that's basically R(Q | D) 

        similarities = similarities * self.gamma3

        similarities = tf.where(tf.equal(masks, True), x=tf.constant(-float('inf'), dtype=tf.float32, shape=masks.shape), y=similarities)   # Punishing masked values with negative infinity

        similaritiesT = tf.transpose(similarities, perm=[1, 0])

        softmax_similarities = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=similarities, labels=label)
        softmax_similaritiesT = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=similaritiesT, labels=label)

        loss0 = tf.reduce_mean(softmax_similarities)
        loss1 = tf.reduce_mean(softmax_similaritiesT)

        return (loss0 + loss1) * self.smooth_lambda

    def sentence_loss(self, image_feature, sentence_vector, class_ids): 
        batch_size = sentence_vector.shape[0]
        label = tf.cast(range(batch_size), tf.int32)

        # Iterate through the batch samples one by one 
        # Mask samples that belong to the same class but are not the same sample. Masked values = 1.
        masks = []
        for i in range(batch_size):
            mask = (class_ids.numpy() == class_ids[i].numpy()).astype(np.uint8)
            mask[i] = 0
            masks.append(np.reshape(mask, newshape=[1, -1]))

        masks = tf.cast(tf.concat(masks, axis=0), tf.float32)

        cnn_code = tf.expand_dims(image_feature, axis=0)
        rnn_code = tf.expand_dims(sentence_vector, axis=0)

        cnn_code_norm = tf.norm(cnn_code, axis=-1, keepdims=True)
        rnn_code_norm = tf.norm(rnn_code, axis=-1, keepdims=True)

        scores0 = tf.matmul(cnn_code, rnn_code, transpose_b=True)
        norm0 = tf.matmul(cnn_code_norm, rnn_code_norm, transpose_b=True)
        scores0 = scores0 / tf.clip_by_value(norm0, clip_value_min=1e-8, clip_value_max=float('inf')) * self.gamma3

        scores0 = tf.squeeze(scores0, axis=0)

        scores0 = tf.where(tf.equal(masks, True), x=tf.constant(-float('inf'), dtype=tf.float32, shape=masks.shape), y=scores0)
        # print('score0======>', scores0)
        scores1 = tf.transpose(scores0, perm=[1, 0])
        # print('label: ', label)
        loss0 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores0, labels=label))
        # print('loss0: ', loss0)
        loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores1, labels=label))

        loss = (loss0 + loss1) / self.smooth_lambda

        return loss

    def attention(self, image_feature, word_vector): 
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