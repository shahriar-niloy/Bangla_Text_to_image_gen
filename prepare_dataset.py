import tensorflow as tf
import sys
import sys
import os
import pickle 
import time 
import numpy as np 
import random
from PIL import Image 
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.getcwd())

from config import config 
# from Components.TextEncoder.Model import TextEncoder
# from Components.ConditionalAugmentation import ConditionalAugmentation
# from Components.GAN.Generator.Model import InitialGeneratorBlock, IntermediateGeneratorBlock
# from Components.GAN.Discriminator import Discriminator_64, Discriminator_128, Discriminator_256

#Mock Dataset
textDataset = ["Something is better than nothing", "The stakes are high so is the reward"]
# Cant have batch dimension


class TextPreprocessor:
    def __init__(self, dictionary_size, unknown_token_symbol):
        self.dictionary_size = dictionary_size
        self.unknown_token_symbol = unknown_token_symbol
        self.sequence_length = config.DATASET['TEXT']['SEQUENCE_LENGTH']

    def tokenize(self, texts):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.dictionary_size, oov_token=self.unknown_token_symbol)
        tokenizer.fit_on_texts(texts)
        return tokenizer

    def preprocess(self, texts):
        tokenizer = self.tokenize(texts)
        text_sequences = tokenizer.texts_to_sequences(texts)
        padded_text_sequences = tf.keras.preprocessing.sequence.pad_sequences(text_sequences, maxlen=self.sequence_length,  padding='post')
        word_index = tokenizer.word_index
        return padded_text_sequences, word_index



class CaptionDataset:
    def __init__(self, textPreprocessor, filenames=[], phase='TRAIN'):
        self.captions = []
        self.filenames = filenames
        self.textPreprocessor = textPreprocessor
        self.captions_per_image = config.DATASET['TEXT']['CAPTIONS_PER_IMAGE']
        self.dataset_path = config.DATASET['DIR']
        self.text_path = config.DATASET['TEXT']['DIR']
        self.filenames_path = config.DATASET[phase]['FILE_NAMES']
        self.checkpoint = config.DATASET['TEXT']['SAVED_CAPTIONS']

    def load(self):
        # load captions from dataset and set it to self.captions
        if os.path.isfile(self.checkpoint):                         # If a caption checkpoint exists, load from the saved file.........................=> Pending
            with open(self.checkpoint, 'rb') as file:
                x = pickle.load(file)
                # Then load from the saved file
        else:                                                       # If no such checkpoint exists, save the progress for later use....................=> Pending
            captions = self.read_captions()
            self.captions = self.preprocess(captions)

    def read_captions(self): 
        all_captions = []
        for i in range(len(self.filenames)):                                     # Iterate through all filenames one by one 
            cap_path = '%s%s.txt' % (self.text_path, self.filenames[i])        # the path of the caption text file 
            with open(cap_path, "r") as f:                      # Open one caption txt file for corresponding image 
                data = f.read()                                 # Read the context of the txt file
                captions = data.split('\n')                     # Get a list of captions from data
                cnt = 0
                for cap in captions:                            # Iterate through the captions list we just got one by one 
                    if len(cap) == 0:                           # Do nothing is the current length of the caption is zero
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")      

                    all_captions.append(cap)
                    cnt += 1
                    if cnt == self.captions_per_image:              
                        break
                if cnt < self.captions_per_image:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def preprocess(self, captions):
        sequence_length = self.textPreprocessor.sequence_length
        preprocessed_captions, self.word_index = self.textPreprocessor.preprocess(captions)
        reshaped_captions = tf.reshape(preprocessed_captions, (-1, self.captions_per_image, sequence_length))
        self.index_word = self.index_to_word(self.word_index)
        return reshaped_captions

    def index_to_word(self, word_index):
        index_word = { y: x for x, y in word_index.items() }
        return index_word

    def get_captions(self):
        return self.captions
    
    def show_text_from_sequence(self, sequence):
        text = ""
        for seq in sequence: 
            if seq.numpy() == 0: 
                break
            text = text + self.index_word[seq.numpy()] + " "
        return text


class ImageDataset: 
    def __init__(self, filenames):
        self.batch_size = config.DATASET['BATCH_SIZE']
        self.filenames = filenames
        self.generator_blocks = config.GENERATOR['NUMBER_OF_BLOCKS']
        initial_generator_image_size = config.GENERATOR['INIT_OUTPUT_IMAGE_SIZE']
        self.image_sizes = [initial_generator_image_size * 2**i for i in range(self.generator_blocks)]
        self.image_directory = config.DATASET['IMAGE']['DIR']
        self.bbox_path = config.DATASET['BOUDNDING_BOX_PATH']
        self.image_file_names_path = config.DATASET['ALL_IMAGE_PATH']
        self.filename_bbox = self.load_bbox()
        self.filename_index = range(0, len(self.filenames))

    def load_image(self, filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3, dct_method='INTEGER_ACCURATE')
        return image

    def load_batch(self, batch_start_index=None, batch_end_index=None, indices=None):
        batch_images = [[], [], []]

        if indices is not None: 
            batch_image_filenames = self.filenames[indices]
        else: 
            if batch_start_index is None or batch_end_index is None:
                batch_image_filenames = self.filenames[0:self.batch_size]    
            else:
                batch_image_filenames = self.filenames[batch_start_index:batch_end_index]

        for i in range(0, self.batch_size):
            image_filename = batch_image_filenames[i]
            image_filepath = self.image_directory + "/" + image_filename + ".jpg"
            image = self.load_image(image_filepath)
            image = self.preprocess(image, image_filename)
            for i in range(self.generator_blocks):
                resized_image = tf.image.resize(image, [self.image_sizes[i], self.image_sizes[i]]) 
                batch_images[i].append(resized_image)

        for i in range(self.generator_blocks):
            batch_images[i] = tf.convert_to_tensor(batch_images[i])

        return batch_images

    def preprocess(self, image, image_filename):
        if self.filename_bbox is not None: 
            bbox = self.filename_bbox[image_filename]
            if bbox is not None:
                try:
                    image = tf.image.crop_to_bounding_box(image, bbox[1], bbox[0], bbox[3], bbox[2])
                except:
                    print("Bounding box crop error: " + image_filename)
        
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def show_image(self, image):
        plt.imshow(image)
        plt.show()

    def load_bbox(self):
        bounding_boxes = pd.read_csv(self.bbox_path, delim_whitespace=True, header=None).astype(int)
        image_filenames = pd.read_csv(self.image_file_names_path, delim_whitespace=True, header=None)
        filenames = image_filenames[1].tolist()                            
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}      
        numImgs = len(filenames)

        for i in range(0, numImgs):
            bbox = bounding_boxes.iloc[i][1:].tolist()               
            key = filenames[i][:-4]
            filename_bbox[key] = bbox

        return filename_bbox

        

class DatasetLoader(tf.keras.utils.Sequence):
    '''
        Caption Dimension: (batch_size, self.captions_per_image, sequence_length)
        Image Dimension: (batch_size, number_of_generator_blocks, width, height, channel)
    '''
    def __init__(self, phase='TRAIN'):
        textPreprocessor = TextPreprocessor(config.TEXT_ENCODER['DICTIONARY_SIZE'], config.TEXT_ENCODER['UNKNOWN_TOKEN_SYMBOL'])
        self.filenames = self.load_filenames(config.DATASET[phase]['FILE_NAMES'])
        self.class_ids = self.load_class_id(config.DATASET[phase]['CLASS_NAMES'])
        self.captionLoader = CaptionDataset(textPreprocessor, self.filenames, phase)
        self.imageLoader = ImageDataset(self.filenames)
        self.batch_size = config.DATASET['BATCH_SIZE']
        self.sequence_length = config.DATASET['TEXT']['SEQUENCE_LENGTH']
        self.dataset_index_list = None

    def load(self): 
        self.captionLoader.load()
        self.captions = self.captionLoader.get_captions()
        self.dataset_index_list = list(range(0, len(self.captions)))
        random.shuffle(self.dataset_index_list)

    def __len__(self):
        return self.captions.shape[0] // self.batch_size

    def __getitem__(self, index):
        batch_start_index = index * self.batch_size
        batch_end_index = (index + 1) * self.batch_size
        captions_per_image = self.captionLoader.captions_per_image
        batch_captions = np.array([])

        for i in range(batch_start_index, batch_end_index):
            random_index = tf.random.uniform([], 0, captions_per_image, tf.dtypes.int32)
            caption_index = self.dataset_index_list[i]

            caption = self.captions[caption_index][random_index]
            batch_captions = np.append(batch_captions, caption, axis=0)

        batch_captions = tf.reshape(batch_captions, (self.batch_size, self.sequence_length))
        batch_indices = self.dataset_index_list[batch_start_index : batch_end_index]
        batch_images = self.imageLoader.load_batch(indices=batch_indices)
        batch_class_ids = tf.convert_to_tensor(self.class_ids[batch_indices])           # Class Ids are not being randomized 
        
        # Test = Iterating through the batch to see if we are getting the right caption and picture pair 
        # for j in range(self.batch_size): 
        #     print('Current filename: ', self.filenames[index + j])
        #     print('Current Caption: ', self.captionLoader.show_text_from_sequence(batch_captions[j]))
        #     print('Current Classes: ', batch_class_ids[j])
        #     self.imageLoader.show_image(batch_images[2][j])

        return batch_captions, batch_images, batch_class_ids

    def load_filenames(self, filepath):
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return np.array(filenames)

    def load_class_id(self, filepath): 
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                class_id = pickle.load(f, encoding="bytes")
        else:
            class_id = np.arange(total_num)

        return np.array(class_id) 

    def get_captions(self): 
        return self.captions 

    def shuffle(self):
        self.dataset_index_list = tf.random.shuffle(self.dataset_index_list)

    def on_epoch_end(self):
        self.shuffle()

# textPreprocessor = TextPreprocessor(dictionary_size, unknown_token_symbol)
# captionDataset = CaptionDataset(textPreprocessor)

# captionDataset.load()
# captionDataset.preprocess()
# captions_tensor = captionDataset.get_captions()
# print('Captions: ', captions_tensor.shape)

# textEncoder = TextEncoder(vocabulary_size=dictionary_size, embedding_dim=16, encoder_units=2, batch_size=20, rnn_type='LSTM')
# word_vector, sentence_vector = textEncoder(captions_tensor)
# print('word vector: ', word_vector.get_shape())
# print('sentence vector: ', sentence_vector.get_shape())

# ca_net = ConditionalAugmentation(4)
# condition_code = ca_net(sentence_vector)
# print('Condition vector: ', condition_code.get_shape())

# noise_vector = tf.random.normal(shape=[2, 4])
# print('Noise: ', noise_vector.get_shape())

# noise_condition_vector = tf.concat([condition_code, noise_vector], axis=-1)
# print('Concatenated vector: ', noise_condition_vector.get_shape())
# # Make a class for Image dataset

# hidden0, image0 = InitialGeneratorBlock()(noise_condition_vector)
# print('Output from the first generator block: ', hidden0.get_shape(), image0.get_shape())

# uncond_logits0, cond_logits0 = Discriminator_64(64)(image0, sentence_vector)
# print('Uncond_logits0: ', uncond_logits0, 'Cond_logits0: ', cond_logits0)

# hidden1, image1 = IntermediateGeneratorBlock(16)(hidden0, word_vector)
# print('Word Context vector: ', hidden1.get_shape(), image1.get_shape())

# uncond_logits1, cond_logits1 = Discriminator_128(64)(image1, sentence_vector)
# print('Uncond_logits1: ', uncond_logits1, 'Cond_logits1: ', cond_logits1)

# hidden2, image2 = IntermediateGeneratorBlock(16)(hidden1, word_vector)
# print('Word Context vector2: ', hidden2.get_shape(), image2.get_shape())

# uncond_logits2, cond_logits2 = Discriminator_256(64)(image2, sentence_vector)
# print('Uncond_logits1: ', uncond_logits2, 'Cond_logits1: ', cond_logits2)

# # Discriminator loss calculation 
# real_logit = uncond_logits0
# fake_logit = uncond_logits0
# cond_real_logit = cond_logits0
# cond_fake_logit = cond_logits0
# cond_wrong_logit = cond_logits0

# loss = discriminator_loss(real_logit, fake_logit, cond_real_logit, cond_fake_logit, cond_wrong_logit)
# print('Discriminator loss: ', loss)

