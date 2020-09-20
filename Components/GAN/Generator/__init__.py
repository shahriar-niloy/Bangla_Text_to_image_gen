import sys
import os
import tensorflow as tf 
import numpy as np 

sys.path.append(os.getcwd())

from config import config 
from Components.Attention.Model import Attention

def upSample(x, scale_factor=2):
    _, height, width, _ = x.get_shape().as_list()
    scaled_size = [height * scale_factor, width * scale_factor]    
    return tf.image.resize(x, size=scaled_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.block1= tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=channels, kernel_size=3, strides=1),
            tf.keras.layers.ReLU(),
        ])

    def call(self, x):
        initial_activation = x
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        x = self.block1(x)
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        x = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=3, strides=1)(x)
        x = x + initial_activation
        x = tf.keras.layers.ReLU()(x)
        return x


# Changes the spatial dimension of the hidden feature
class UpBlock(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(UpBlock, self).__init__()
        self.conv2d = tf.keras.layers.Conv2D(filters=channels, kernel_size=3, strides=1, padding='same')

    def call(self, x):
        x = upSample(x)
        x = self.conv2d(x)
        return x

# Changes the channel dimesion while keeping the spatial dimension same
class ChangeChannel(tf.keras.layers.Layer):
    def __init__(self, output_channel):
        super(ChangeChannel, self).__init__()
        self.output_channel = output_channel
        self.conv = tf.keras.layers.Conv2D(filters=self.output_channel, kernel_size=3, strides=1)

    def call(self, x):
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        x = self.conv(x)
        return x 


class GenerateImage(tf.keras.layers.Layer):
    def __init__(self):
        super(GenerateImage, self).__init__()
        self.image_generator = ChangeChannel(3)                                 # 3 because RGB image has three channels
        self.tanh = tf.keras.layers.Activation(tf.keras.activations.tanh)

    def call(self, x):
        image = self.image_generator(x)
        image = self.tanh(image)
        return image 

# Dynamically create generator blocks. Take the number of generator blocks from config file
class InitialGeneratorBlock(tf.keras.layers.Layer):
    def __init__(self, initial_channel, name="Initial_generator_block"):
        super(InitialGeneratorBlock, self).__init__()
        self.initial_channel = initial_channel
        self.starting_width_height = 4 
        self.init_gen_dim = self.initial_channel * self.starting_width_height**2
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.init_gen_dim),
            tf.keras.layers.Reshape((self.starting_width_height, self.starting_width_height, self.initial_channel)),
            UpBlock(self.initial_channel/2),
            UpBlock(self.initial_channel/4),
            UpBlock(self.initial_channel/8),
            UpBlock(self.initial_channel/16)
        ])
        self.image_generator = GenerateImage()

    def call(self, x):
        x = self.model(x)
        image = self.image_generator(x)
        return x, image


class IntermediateGeneratorBlock(tf.keras.layers.Layer):
    def __init__(self, output_channels, name="IntermediateGeneratorBlock"):
        super(IntermediateGeneratorBlock, self).__init__()
        self.attention = Attention(common_dimension=16)
        self.res1 = ResidualBlock(output_channels * 2)
        self.res2 = ResidualBlock(output_channels * 2)
        self.upBlock = UpBlock(output_channels)
        self.image_generator = GenerateImage()

    def call(self, hidden, word_vectors):
        ''' 
            hidden: Output from the previos block
                    Dimension: [batch, height, width, channel]
            word_vectors: Word vectors from RNN text encoder for attention network
                    Dimension: [batch, seq_len, number of direction * RNN units]
        '''
        batch, height, width, channel = hidden.get_shape()
        # Word_context = [batch, common_dimension, sub_regions]
        word_context = self.attention(hidden, word_vectors)
        # hidden = transpose operation => [batch, sub_regions, common_dimension]
        word_context = tf.transpose(word_context, perm=[0,2,1])
        # word_context = reshape => [batch, height, width, common_dimension]
        word_context = tf.reshape(word_context, shape=[batch, height, width, channel])
        # hidden_word_context = Concat Operation => [batch, height, weight, common_dim*2]
        hidden_word_context = tf.concat([hidden, word_context], axis=-1)
        print('hidden_word_context shape: ',hidden_word_context.get_shape())
        # x = self.res1(hidden_word_context)
        x = self.res1(hidden_word_context)
        x = self.res2(x)
        x = self.upBlock(x)

        image = self.image_generator(x)
        return x, image


class Generator(tf.keras.Model):
    def __init__(self, name="Generator"):
        super(Generator, self).__init__()
        self.initial_channel = 256
        self.gamma2 = 5.0
        self.gamma3  = config.DAMSM['SENTENCE_LOSS_CONSTANT']
        self.smooth_lambda = config.SMOOTH['LAMBDA']
        self.learning_rate = config.MODEL['LEARNING_RATE']
        self.build_model()
        pass

    def build_model(self):
        self.block0 = InitialGeneratorBlock(self.initial_channel)
        self.block1 = IntermediateGeneratorBlock(self.initial_channel/16)
        self.block2 = IntermediateGeneratorBlock(self.initial_channel/16)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.5, beta_2=0.999, epsilon=1e-08)

    def forward(self, noise_condition_vector, word_vector):
        hidden0, img0 = self.block0(noise_condition_vector)
        hidden1, img1 = self.block1(hidden0, word_vector)
        hidden2, img2 = self.block2(hidden1, word_vector)
        return hidden0, img0, hidden1, img1, hidden2, img2

    def loss(self, uncond_fake_logits, cond_fake_logits):
        batch = uncond_fake_logits.shape[0]
        unconditional_fake_loss = tf.keras.losses.BinaryCrossentropy()(tf.ones([batch]), uncond_fake_logits)
        conditional_fake_loss = tf.keras.losses.BinaryCrossentropy()(tf.ones([batch]), cond_fake_logits)
        return unconditional_fake_loss + conditional_fake_loss

    def train(self, discriminator, image_encoder, damsm_attention, fake_images, word_vector, global_sentence_vector, condition_vector, class_ids, condAugmentation): 
        # with tf.GradientTape() as generator_tape:    
        fake_img0, fake_img1, fake_img2 = fake_images

        # Feed fake images to discriminator and say it is real 
        fake_uncond_logits0, fake_cond_logits0 = discriminator.block0(fake_img0, condition_vector)
        fake_uncond_logits1, fake_cond_logits1 = discriminator.block1(fake_img1, condition_vector)
        fake_uncond_logits2, fake_cond_logits2 = discriminator.block2(fake_img2, condition_vector)

        # Calculate total generator loss 
        loss0 = self.loss(fake_uncond_logits0, fake_cond_logits0)
        loss1 = self.loss(fake_uncond_logits1, fake_cond_logits1)
        loss2 = self.loss(fake_uncond_logits2, fake_cond_logits2)

        generator_loss = loss0 + loss1 + loss2

        # Word loss for the last generator 
        sub_region_vector, global_image_vector = image_encoder(fake_img2)
        print('Sub region vector: ', sub_region_vector.shape)
        word_loss = self.word_loss(damsm_attention, sub_region_vector, word_vector, class_ids)                       # FAKE Class_ids given 

        # Sentence loss for the last generator 
        sentence_loss = self.sentence_loss(global_image_vector, global_sentence_vector, class_ids)

        total_loss = generator_loss + word_loss + sentence_loss                                                      # NEED: Right implementation of total loss 

        # g_train_variable = self.block0.trainable_variables + self.block1.trainable_variables + self.block2.trainable_variables #+ condAugmentation.trainable_variables
        
        # # print('G-train var=============================================================>: ', g_train_variable)
        # # print('g block vart============================================================>', self.block0.trainable_variables)
        # # print([var.name for var in generator_tape.watched_variables()])
        # # print(self.block0.trainable_variables)

        # g_gradient = generator_tape.gradient(total_loss, g_train_variable)
        
        # # print('Gradients: -------------------------------------------------------------->', g_gradient)

        # for var, grad in zip(g_train_variable, g_gradient):
        #     print(f'{var.name} = {grad}')

        # # self.optimizer.apply_gradients(zip(g_gradient, g_train_variable))

        return total_loss

    def word_loss(self, damsm_attention, image_feature, word_vector, class_ids):
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
            
            print('Image Feature: ', image_feature.shape)

            # Get region context vectors using DAMSM Attention. 
            region_context = damsm_attention(image_feature, word)           # Dimension [bs, ndf, seq_len]
            print('Region context vector: ', region_context.shape)

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
        print('score0======>', scores0)
        scores1 = tf.transpose(scores0, perm=[1, 0])
        print('label: ', label)
        loss0 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores0, labels=label))
        print('loss0: ', loss0)
        loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores1, labels=label))

        loss = (loss0 + loss1) / self.smooth_lambda

        return loss 


def cosine_similarity(x, y):                    # --> batch_size*words_num x nef
    xy = tf.reduce_sum(x * y, axis=-1)          # Sum reduce along dimension  1 or row dimension 
    x = tf.norm(x, axis=-1)                     # Summation of x1^2 and then square root the summation. Do this along the row dimension as specified by axis = -1
    y = tf.norm(y, axis=-1)                     # This is getting the magnitude 

    similarity = (xy / ((x * y) + 1e-8))        # Apply the cosine similarity formula 

    return similarity                           # A vector of size batch_size * number_of_words 

# Generator train steps: 
    # Feed the discriminator fake image and say it is real 
    # Get the prediction from discriminator 
    # Calculate the generator loss for both condition and uncondition 

# At the last stage of the discriminator: 
    # Calculate the word loss 
    # Calculate the sentence loss 

# def DAMSM_loss(): 
# # DAMSM Loss 
#     # Calculate word loss
#         # Description, given image
#         # Image, given description 

#     # Calculate Sentence loss
#         # Description, given image 
#         # Image, given description 
#     word_loss0, word_loss1 = word_loss()
#     sentence_loss0, sentence_loss1 = sentence_loss()

#     damsm_loss = (word_loss0 + word_loss1 + sentence_loss0 + sentence_loss1) * config.DAMSM['DAMSM_LOSS_CONSTANT']

#     return damsm_loss