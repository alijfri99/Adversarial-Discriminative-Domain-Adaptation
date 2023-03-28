from Extractors.LabeledDatasetExtractor import LabeledDatasetExtractor
from Processors.ImageProcessor import ImageProcessor
from Datasets.LabeledDataset import LabeledDataset
from torchvision import transforms
from Networks.Encoders.LeNetEncoder import LeNetEncoder
from Networks.Discriminators.LeNetDiscriminator import LeNetDiscriminator
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Experiments.Experiment import Experiment
from Networks.Encoders.Identity import Identity
from Networks.Encoders.VGG16Encoder import VGG16Encoder
from Networks.Classifiers.VGG16Classifier import VGG16Classifier
from Networks.Discriminators.VGG16Discriminator import VGG16Discriminator
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import numpy
import os

'''extractor = LabeledDatasetExtractor(ImageProcessor(), 60000, (28, 28))
extractor.extract('Data/MNIST/training')
extractor.save('Code/StoredDatasets/MNIST', 'mnist_training')

extractor = LabeledDatasetExtractor(ImageProcessor(), 10000, (28, 28))
extractor.extract('Data/MNIST/testing')
extractor.save('Code/StoredDatasets/MNIST', 'mnist_testing')'''

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

nyurgb = LabeledDataset('/content/drive/MyDrive/StoredDatasets/NYUD2', 'nyud2_training', transforms.ToTensor())
nyudepth = LabeledDataset('/content/drive/MyDrive/StoredDatasets/NYUD2', 'nyud2_testing', transforms.ToTensor())
print(numpy.unique(nyurgb.labels))
print(numpy.min(nyurgb.labels))
print(numpy.max(nyurgb.labels))

source_encoder = VGG16Encoder().to(device)
target_encoder = VGG16Encoder().to(device)
classifier = VGG16Classifier().to(device)
discriminator = VGG16Discriminator().to(device)
batch_size = 128
num_iterations = 20000
classification_criterion = nn.CrossEntropyLoss()
epochs = 10

classification_lr = 0.001
discriminator_lr = target_encoder_lr = 0.0002

print("Starting the experiment")

experiment = Experiment(nyurgb, nyudepth, source_encoder, target_encoder, classifier, discriminator, epochs, batch_size, classification_criterion,
                        classification_lr, discriminator_lr, target_encoder_lr, num_iterations, device)
experiment.run()
