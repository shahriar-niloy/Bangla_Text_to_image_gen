from model import Model 
from prepare_dataset import DatasetLoader

datasetLoader = DatasetLoader()

model = Model(datasetLoader)

print(model.captions.shape)

model.train(5)
