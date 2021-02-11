import sys
import os

sys.path.append(os.getcwd())

from model import Model 
from prepare_dataset import DatasetLoader

from Components.DAMSM import DAMSM

datasetLoader = DatasetLoader()

# Pretrain DAMSM
# damsm = DAMSM(datasetLoader)
# damsm(3)

model = Model(datasetLoader)
model.train(1)
