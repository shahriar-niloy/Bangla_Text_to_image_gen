import sys
import os
import tensorflow as tf 
import time 
import numpy as np 

sys.path.append(os.getcwd())

from config import config 
from Components.GAN.Generator import Generator
from Components.GAN.Discriminator import Discriminator
from Components.ConditionalAugmentation import ConditionalAugmentation
from Components.TextEncoder.Model import TextEncoder
from Components.ImageEncoder import ImageEncoder
from Components.DAMSMAttention import DAMSMAttention

class Model: 
    def __init__(self, datasetLoader):
        self.datasetLoader = datasetLoader
        self.conditionalDimension = config.CONDITIONING_AUGMENTATION['DIMENSION']
        self.dictionary_size = config.TEXT_ENCODER['DICTIONARY_SIZE']
        self.embedding_dim = config.TEXT_ENCODER['EMBEDDING_DIMENSION']
        self.encoder_units = config.TEXT_ENCODER['ENCODER_UNITS']
        self.batch_size = config.DATASET['BATCH_SIZE']
        self.checkpoint_path = config.MODEL['CHECKPOINT_PATH']
        self.max_numer_of_checkpoints = config.MODEL['MAX_NUMBER_OF_CHECKPOINTS']
        self.load_data()
        self.build_model()

    def load_data(self):
        self.datasetLoader.load()
        self.captions = self.datasetLoader.get_captions()

    def build_model(self):
        self.image_encoder = ImageEncoder(self.embedding_dim)
        self.textEncoder = TextEncoder(self.dictionary_size, self.embedding_dim, self.encoder_units, self.batch_size)
        self.condAugmentation = ConditionalAugmentation(self.conditionalDimension)
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.damsm_attention = DAMSMAttention()

    def train(self, epochs=5):
        batch = self.captions.shape[0]
        number_of_step = batch // self.batch_size

        self.checkpoint = tf.train.Checkpoint(text_encoder=self.textEncoder, image_encoder=self.image_encoder, conditioning_augmentation=self.condAugmentation, generator=self.generator, discriminator=self.discriminator)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_path, max_to_keep=self.max_numer_of_checkpoints)

        if self.checkpoint_manager.latest_checkpoint is not None:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

        for current_epoch in range(epochs):
            dataset_iter = iter(self.datasetLoader)

            for current_step in range(number_of_step):
                batch_captions, batch_images, class_ids = next(dataset_iter)
                real_image0, real_image1, real_image2 = batch_images
                # print('Batch captinos: ', batch_captions, ' shape: ', batch_captions.shape)
                # print('Batch images: ', batch_images, ' shape: ', np.array(batch_images).shape)
                print('Batch captinos shape: ', batch_captions.shape)
                print('Batch images shape: ', np.array(batch_images).shape)
                # time.sleep(5)    # pausing here to check if the captions and data are being loaded correctly or not 
                print('NEW BATCH GENERATED===============================================>')

                word_vector, sentence_vector = self.textEncoder(batch_captions)

                print('word vector: ', word_vector.shape, ' sentence vector: ', sentence_vector.shape)

                # Apply conditional augmentation
                condition_vector, mean, std = self.condAugmentation(sentence_vector)

                # Concat noise vector with condition code
                noise_vector = tf.random.normal(shape=[self.batch_size, self.conditionalDimension])
                print('condition vector ', condition_vector.shape, ' noise_vector: ', noise_vector.shape)
                noise_condition_vector = tf.concat([condition_vector, noise_vector], axis=-1)
                
                # Generator Forward move
                generator_features = self.generator.forward(noise_condition_vector, word_vector)
                hidden0, fake_img0, hidden1, fake_img1, hidden2, fake_img2 = generator_features

                # Train Discriminator 
                discriminator_loss = self.discriminator.train([real_image0, real_image1, real_image2], [fake_img0, fake_img1, fake_img2], condition_vector)

                # Train Generator => Give the processed batched data to generator and it will generate the fake images necessary. Also make the attention layer a part of it. 
                generator_loss = self.generator.train(self.discriminator, self.image_encoder, self.damsm_attention, [fake_img0, fake_img1, fake_img2], word_vector, sentence_vector, condition_vector, class_ids, self.condAugmentation)

                # Calculate KL Loss
                kl_loss = self.condAugmentation.kl_loss(mean, std)

                # Penalize generator by adding KL_loss
                generator_loss = generator_loss + kl_loss

                # print('Generator features: ', generator_features.get_shape())
                print('Discriminator loss: ', discriminator_loss)
                print('Generator loss: ', generator_loss)

            self.checkpoint_manager.save()

    def test(self): 
        pass 

