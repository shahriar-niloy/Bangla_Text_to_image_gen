import os 

dir_name = os.path.dirname(__file__) + '/..'

print('Dir name: ', dir_name)

DATASET = {
  'BATCH_SIZE': 5,
  'DIR': dir_name + '/Dataset/',
  'BOUDNDING_BOX_PATH': dir_name + '/Dataset/bounding_boxes.txt',
  'ALL_IMAGE_PATH': dir_name + '/Dataset/images.txt',
  'TRAIN': {
    'DIR': dir_name + '/Dataset/train/',
    'FILE_NAMES': dir_name + '/Dataset/train/filenames.pickle',
    'CLASS_NAMES': dir_name + '/Dataset/train/class_info.pickle'
  }, 
  'TEST': {
    'DIR': dir_name + '/Dataset/test/',
    'FILE_NAMES': dir_name + '/Dataset/test/filenames.pickle',
    'CLASS_NAMES': dir_name + '/Dataset/test/class_info.pickle'
  },
  'IMAGE': {
    'DIR': dir_name + '/Dataset/images/'
  },
  'TEXT': {
    'DIR': dir_name + '/Dataset/text/',
    'SEQUENCE_LENGTH': 20,
    'CAPTIONS_PER_IMAGE': 10,
    'SAVED_CAPTIONS': dir_name + '/Checkpoint/Captions/captions.pickle'
  },
}

TEXT_ENCODER = {
  'RNN_TYPE': 'LSTM',  # 'GRU',
  'DICTIONARY_SIZE': 10000,
  'EMBEDDING_DIMENSION': 16,
  'ENCODER_UNITS': 8,                   # Should be half of embedding dimension 
  'UNKNOWN_TOKEN_SYMBOL': '<OOV>'
}

GENERATOR = {
  'NUMBER_OF_BLOCKS': 3,
  'INIT_OUTPUT_IMAGE_SIZE': 64
}

DISCRIMINATOR = {
  'DIMENSION': 16
}

MODEL = {
  'NOISE_DIMENSION': 10,
  'CHECKPOINT_PATH': dir_name + 'Checkpoint',
  'MAX_NUMBER_OF_CHECKPOINTS': 2,
  'LEARNING_RATE': 0.05
}

SMOOTH = {
  'LAMBDA': 5.0
}

DAMSM = {
  'DAMSM_LOSS_CONSTANT': 0.2,
  'SENTENCE_LOSS_CONSTANT': 10.0,
  'GAMMA1': 4.0
}

CONDITIONING_AUGMENTATION = {
  'DIMENSION': 10
}