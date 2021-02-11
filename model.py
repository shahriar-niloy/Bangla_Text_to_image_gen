import sys
import os
import tensorflow as tf 
import time 
import numpy as np 

sys.path.append(os.getcwd())

from config import config 
from Components.GAN.Generator import Generator
from Components.GAN.Discriminator import Discriminator
from Components.TextEncoder.Model import TextEncoder
from Components.ImageEncoder import ImageEncoder
from Components.DAMSMAttention import DAMSMAttention

class Model: 
    def __init__(self, datasetLoader):
        self.datasetLoader = datasetLoader
        self.dictionary_size = config.TEXT_ENCODER['DICTIONARY_SIZE']
        self.embedding_dim = config.TEXT_ENCODER['EMBEDDING_DIMENSION']
        self.encoder_units = config.TEXT_ENCODER['ENCODER_UNITS']
        self.batch_size = config.DATASET['BATCH_SIZE']
        self.model_checkpoint_path = config.MODEL['CHECKPOINT_PATH']
        self.encoder_checkpoint_path = config.DAMSM['CHECKPOINT_PATH']
        self.model_max_numer_of_checkpoints = config.MODEL['MAX_NUMBER_OF_CHECKPOINTS']
        self.encoder_max_numer_of_checkpoints = config.DAMSM['MAX_NUMBER_OF_CHECKPOINTS']
        self.save_interval = config.MODEL['SAVE_INTERVAL']
        self.load_data()
        self.build_model()

    def load_data(self):
        self.datasetLoader.load()
        self.captions = self.datasetLoader.get_captions()

    def build_model(self):
        self.image_encoder = ImageEncoder(self.embedding_dim)
        self.textEncoder = TextEncoder(self.dictionary_size, self.embedding_dim, self.encoder_units, self.batch_size)
        # self.condAugmentation = ConditionalAugmentation(self.conditionalDimension)
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.damsm_attention = DAMSMAttention()

    def train(self, epochs=5):
        batch = self.captions.shape[0]
        number_of_step = batch // self.batch_size

        self.encoder_checkpoint = tf.train.Checkpoint(text_encoder=self.textEncoder, image_encoder=self.image_encoder)
        self.encoder_checkpoint_manager = tf.train.CheckpointManager(self.encoder_checkpoint, self.encoder_checkpoint_path, max_to_keep=self.encoder_max_numer_of_checkpoints)

        self.model_checkpoint = tf.train.Checkpoint(generator=self.generator, discriminator=self.discriminator)
        self.model_checkpoint_manager = tf.train.CheckpointManager(self.model_checkpoint, self.model_checkpoint_path, max_to_keep=self.model_max_numer_of_checkpoints)

        if self.model_checkpoint_manager.latest_checkpoint is not None:
            print("Loading from model checkpoint...")
            self.model_checkpoint.restore(self.model_checkpoint_manager.latest_checkpoint)
        
        if self.encoder_checkpoint_manager.latest_checkpoint is not None:
            print("Loading from encoder checkpoint...")
            self.encoder_checkpoint.restore(self.encoder_checkpoint_manager.latest_checkpoint)

        for current_epoch in range(epochs):
            dataset_iter = iter(self.datasetLoader)

            for current_step in range(number_of_step):
                batch_captions, batch_images, class_ids = next(dataset_iter)
                real_image0, real_image1, real_image2 = batch_images
                
                # print('Batch captinos: ', batch_captions[0], ' shape: ', batch_captions.shape)
                # print('Batch images: ', batch_images, ' shape: ', np.array(batch_images).shape)
                # print('Batch captinos shape: ', batch_captions.shape)
                # print('Batch images shape: ', np.array(batch_images).shape)
                # print(self.datasetLoader.captionLoader.show_text_from_sequence(batch_captions[0]))
                # self.datasetLoader.imageLoader.show_image(real_image2[0])
                # time.sleep(5)    # pausing here to check if the captions and data are being loaded correctly or not 

                word_vector, sentence_vector = self.textEncoder(batch_captions)

                # Train Discriminator 
                discriminator_loss = self.discriminator.train(self.generator, [real_image0, real_image1, real_image2], sentence_vector, word_vector)

                # Train Generator => Give the processed batched data to generator and it will generate the fake images necessary. Also make the attention layer a part of it. 
                generator_loss = self.generator.train(self.discriminator, self.image_encoder, self.damsm_attention, word_vector, sentence_vector, class_ids)

                # print('Generator features: ', generator_features.get_shape())
                print("Current Step: ", current_step, " Generator Loss: ", generator_loss, " Discriminator loss: ", discriminator_loss, "total loss: ", generator_loss + discriminator_loss)

                if(current_step % self.save_interval == 0):
                    print("Saving to checkpoint...")
                    self.model_checkpoint_manager.save()
                

            self.model_checkpoint_manager.save()

    def test(self): 
        pass 

