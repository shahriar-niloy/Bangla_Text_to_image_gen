import tensorflow as tf
import sys
import os
sys.path.append(os.getcwd())
from Components.GAN.Layers import BatchNorm
from config import config 
import numpy as np 

def conv_spatial_same(out_channel):
    return tf.keras.layers.Conv2D(out_channel, kernel_size=3, strides=1, padding="same")

def conv_spatial_same_leaky_relu(out_channel):
    return tf.keras.Sequential([
        conv_spatial_same(out_channel),
        BatchNorm(),
        tf.keras.layers.LeakyReLU(alpha=0.3)
    ])

class DownScaleByTwo(tf.keras.layers.Layer):
    def __init__(self, output_channel):
        super(DownScaleByTwo, self).__init__()
        self.output_channel = output_channel

    def call(self, x):
        pad_top = pad_bottom = pad_left = pad_right = 1
        x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
        enocoded_data = tf.keras.layers.Conv2D(self.output_channel, kernel_size=4, strides=2)(x)
        return enocoded_data

# Downscale the spatial dimension by n. Genearate a number of ConvLayers to do that 
class EncodeBy16(tf.keras.layers.Layer):
    def __init__(self, channel):
        super(EncodeBy16, self).__init__()
        alpha = 0.2            
        self.model = tf.keras.Sequential([              # Increases the channel 8x and decreases spatial dimension 16x 
            DownScaleByTwo(channel),                 # Downscale spatial dimension by two while increasing the channel by two
            tf.keras.layers.LeakyReLU(alpha=alpha),
            DownScaleByTwo(channel*2),                 # Downscale spatial dimension by two while increasing the channel by two
            BatchNorm(),
            tf.keras.layers.LeakyReLU(alpha=alpha),
            DownScaleByTwo(channel*4),                 # Downscale spatial dimension by two while increasing the channel by two
            BatchNorm(),
            tf.keras.layers.LeakyReLU(alpha=alpha),
            DownScaleByTwo(channel*8),                 # Downscale spatial dimension by two while increasing the channel by two
            BatchNorm(),
            tf.keras.layers.LeakyReLU(alpha=alpha),
        ])

    def call(self, x):
        x = self.model(x)
        return x 

def out_logits():
    return tf.keras.Sequential([
            tf.keras.layers.Conv2D(1, kernel_size=4, strides=4),
            tf.keras.layers.Activation('sigmoid')
        ])

# Takes image of size 64x64, encodes it and classifies it as fake or real 
class Discriminator_64(tf.keras.layers.Layer):
    def __init__(self, channel, name="Discriminator_64"):
        super(Discriminator_64, self).__init__()
        self.encode_16 = EncodeBy16(channel)
        self.joint_conv = conv_spatial_same_leaky_relu(channel*8)
        self.conditional_logits = out_logits()
        self.unconditional_logits = out_logits()

    def build(self, input_shape):
        self.batch = len(input_shape)                                                     # Set batch dimension at runtime
        print('DiscBatch: ', self.batch)

    def call(self, x, condition):                   # Here condition is the sentence vector
        encoded_image = self.encode_16(x)                       # input_channel*8 x input_height/16 x input_width/16
        # Join the condition with encoded_image
        cond_batch, cond_feature_dim = condition.get_shape()
        condition = tf.reshape(condition, [cond_batch, 1, 1, cond_feature_dim])
        condition = tf.tile(condition, [1, 4, 4, 1]) 
        encoded_image_condition = tf.concat([encoded_image, condition], axis=-1)          # Concat encoded image with condition in the last dimension, that is the channel dimension
        print('Encoded Image Condition: ', encoded_image_condition.get_shape())
        encoded_image_condition = self.joint_conv(encoded_image_condition)
        print('After joinnt conv: ', encoded_image_condition.get_shape())
        # Get unconditional logits 
        uncond_logits = self.unconditional_logits(encoded_image)
        uncond_logits = tf.reshape(uncond_logits, [self.batch])
        # Get conditional logits 
        cond_logits = self.conditional_logits(encoded_image_condition)
        cond_logits = tf.reshape(cond_logits, [self.batch])
        # Return both logits 
        return uncond_logits, cond_logits


class Discriminator_128(tf.keras.layers.Layer):
    def __init__(self, channel, name="Discriminator_128"):
        super(Discriminator_128, self).__init__()
        self.encode_16 = EncodeBy16(channel)
        self.encode_32 = DownScaleByTwo(channel*16)
        self.encode_32_reduced_channel = conv_spatial_same_leaky_relu(channel*8)
        self.joint_conv = conv_spatial_same_leaky_relu(channel*8)
        self.conditional_logits = out_logits()
        self.unconditional_logits = out_logits()

    def build(self, input_shape): 
        self.batch = len(input_shape)                                                     # Set batch dimension at runtime 

    def call(self, x, condition):                   # Here condition is the sentence vector
        # Encode the input image
        encoded_image = self.encode_16(x)                       # input_channel*8 x input_height/16 x input_width/16
        encoded_image = self.encode_32(encoded_image)
        encoded_image = self.encode_32_reduced_channel(encoded_image)
        # Join the condition with encoded_image
        cond_batch, cond_feature_dim = condition.get_shape()
        condition = tf.reshape(condition, [cond_batch, 1, 1, cond_feature_dim])
        condition = tf.tile(condition, [1, 4, 4, 1]) 
        encoded_image_condition = tf.concat([encoded_image, condition], axis=-1)          # Concat encoded image with condition in the last dimension, that is the channel dimension
        print('Encoded Image Condition128: ', encoded_image_condition.get_shape())
        encoded_image_condition = self.joint_conv(encoded_image_condition)
        print('After joinnt conv: ', encoded_image_condition.get_shape())
        # Get unconditional logits 
        uncond_logits = self.unconditional_logits(encoded_image)
        uncond_logits = tf.reshape(uncond_logits, [self.batch])
        # Get conditional logits 
        cond_logits = self.conditional_logits(encoded_image_condition)
        cond_logits = tf.reshape(cond_logits, [self.batch])
        # Return both logits 
        return uncond_logits, cond_logits


class Discriminator_256(tf.keras.layers.Layer):
    def __init__(self, channel, name="Discriminator_256"):
        super(Discriminator_256, self).__init__()
        self.encode_16 = EncodeBy16(channel)
        self.encode_32 = DownScaleByTwo(channel*16)
        self.encode_64 = DownScaleByTwo(channel*32)
        self.encode_64_reduced_channel = tf.keras.Sequential([conv_spatial_same_leaky_relu(channel*16), conv_spatial_same_leaky_relu(channel*8)])
        self.joint_conv = conv_spatial_same_leaky_relu(channel*8)
        self.conditional_logits = out_logits()
        self.unconditional_logits = out_logits()

    def build(self, input_shape): 
        self.batch = len(input_shape)                                                     # Set batch dimension at runtime 

    def call(self, x, condition):                   # Here condition is the sentence vector
        # Encode the input image
        encoded_image = self.encode_16(x)                       # input_channel*8 x input_height/16 x input_width/16
        encoded_image = self.encode_32(encoded_image)
        encoded_image = self.encode_64(encoded_image)
        encoded_image = self.encode_64_reduced_channel(encoded_image)
        # Join the condition with encoded_image
        cond_batch, cond_feature_dim = condition.get_shape()
        condition = tf.reshape(condition, [cond_batch, 1, 1, cond_feature_dim])
        condition = tf.tile(condition, [1, 4, 4, 1]) 
        encoded_image_condition = tf.concat([encoded_image, condition], axis=-1)          # Concat encoded image with condition in the last dimension, that is the channel dimension
        print('Encoded Image Condition256: ', encoded_image_condition.get_shape())
        encoded_image_condition = self.joint_conv(encoded_image_condition)
        print('After joinnt conv: ', encoded_image_condition.get_shape())
        # Get unconditional logits 
        uncond_logits = self.unconditional_logits(encoded_image)
        uncond_logits = tf.reshape(uncond_logits, [self.batch])
        # Get conditional logits 
        cond_logits = self.conditional_logits(encoded_image_condition)
        cond_logits = tf.reshape(cond_logits, [self.batch])
        # Return both logits 
        return uncond_logits, cond_logits

class Discriminator(tf.keras.Model):
    def __init__(self, name="Discriminator"):
        super(Discriminator, self).__init__()
        self.channel = config.DISCRIMINATOR['DIMENSION']                  # This will multiplied by 8 will be the output from the discriminator layer 
        self.learning_rate = config.MODEL['LEARNING_RATE']
        self.build_model()

    def build_model(self):
        self.block0 = Discriminator_64(self.channel)
        self.block1 = Discriminator_128(self.channel)
        self.block2 = Discriminator_256(self.channel)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, )

    def train(self, real_images, fake_images, sentence_vector): 
        with tf.GradientTape() as discriminator_tape:
            batch_size = np.array(real_images).shape[0]

            real_image0, real_image1, real_image2 = real_images
            fake_image0, fake_image1, fake_image2 = fake_images

            # Train with real images 
            real_uncond_logits0, real_cond_logits0 = self.block0(real_image0, sentence_vector)
            real_uncond_logits1, real_cond_logits1 = self.block1(real_image1, sentence_vector)
            real_uncond_logits2, real_cond_logits2 = self.block2(real_image2, sentence_vector)

            # Train with fake images 
            fake_uncond_logits0, fake_cond_logits0 = self.block0(fake_image0, sentence_vector)
            fake_uncond_logits1, fake_cond_logits1 = self.block1(fake_image1, sentence_vector)
            fake_uncond_logits2, fake_cond_logits2 = self.block2(fake_image2, sentence_vector)

            # Train with wrong image caption pair 
            _, wrong_cond_logits0 = self.block0(real_image0, sentence_vector)
            _, wrong_cond_logits1 = self.block1(real_image1, sentence_vector)
            _, wrong_cond_logits2 = self.block2(real_image2, sentence_vector)

            # Calcualte loss for real and fake images
            loss0 = self.calculate_loss(real_uncond_logits0, fake_uncond_logits0, real_cond_logits0, fake_cond_logits0, wrong_cond_logits0)
            loss1 = self.calculate_loss(real_uncond_logits1, fake_uncond_logits1, real_cond_logits1, fake_cond_logits1, wrong_cond_logits1)
            loss2 = self.calculate_loss(real_uncond_logits2, fake_uncond_logits2, real_cond_logits2, fake_cond_logits2, wrong_cond_logits2)

            total_loss = loss0 + loss1 + loss2
        
        discriminator_variables = self.block0.trainable_variables + self.block1.trainable_variables + self.block2.trainable_variables
        discriminator_tape.gradient(total_loss, discriminator_variables)
        # apply graidients to optimizer
        return total_loss

    def calculate_loss(self, real_logits, fake_logits, cond_real_logits, cond_fake_logits, cond_wrong_logits): 
        # Unconditinal Loss:
        batch = real_logits.get_shape()[0]
        # Calculate loss for real image
        unconditional_real_loss = tf.keras.losses.BinaryCrossentropy()(tf.ones([batch]), real_logits)
        # calculate loss for fake image 
        unconditional_fake_loss = tf.keras.losses.BinaryCrossentropy()(tf.zeros([batch]), fake_logits)

        # Conditional Loss:
        # Calculate loss for real image with real caption
        conditional_real_loss = tf.keras.losses.BinaryCrossentropy()(tf.ones([batch]), cond_real_logits)
        # Calculate loss for fake with with real caption 
        conditional_fake_loss = tf.keras.losses.BinaryCrossentropy()(tf.zeros([batch]), cond_fake_logits)
        # Calculate loss for real image with wrong caption
        conditional_wrong_loss = tf.keras.losses.BinaryCrossentropy()(tf.zeros([batch]), cond_wrong_logits)

        # calculate total loss
        total_discriminator_loss = ((unconditional_real_loss + conditional_real_loss) / 2.0) + ((unconditional_fake_loss + conditional_fake_loss + conditional_wrong_loss) / 3.0)

        return total_discriminator_loss





