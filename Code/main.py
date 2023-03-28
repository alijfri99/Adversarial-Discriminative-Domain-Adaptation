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

'''svhn = LabeledDataset('Code/StoredDatasets/SVHN', 'svhn_training', transforms.Compose([transforms.ToTensor()]))
while True:
    inp = int(input("Enter: "))
    plt.imshow(svhn[inp][0][0], cmap='gray')
    plt.show()'''

svhn = LabeledDataset('Code/StoredDatasets/SVHN', 'svhn_training', transforms.Compose([transforms.ToTensor()]))
mnist = LabeledDataset('Code/StoredDatasets/MNIST', 'mnist_training', transforms.Compose([transforms.ToTensor()]))

source_encoder = LeNetEncoder()
target_encoder = LeNetEncoder()
classifier = nn.Linear(500, 10)
discriminator = LeNetDiscriminator()
batch_size = 128
num_iterations = 10000
classification_criterion = nn.CrossEntropyLoss()

classification_lr = 0.001
discriminator_lr = target_encoder_lr = 0.0002

experiment = Experiment(svhn, mnist, source_encoder, target_encoder, classifier, discriminator, batch_size, classification_criterion,
                        classification_lr, discriminator_lr, target_encoder_lr, num_iterations, 'cpu')
experiment.run()