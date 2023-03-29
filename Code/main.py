from Extractors.LabeledDatasetExtractor import LabeledDatasetExtractor
from Processors.ImageProcessor import ImageProcessor
from Datasets.LabeledDataset import LabeledDataset
from torchvision import transforms
from Networks.Encoders.LeNetEncoder import LeNetEncoder
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Experiments.Experiment import Experiment
from Networks.Encoders.Identity import Identity
from Networks.Encoders.VGG16Encoder import VGG16Encoder
from Networks.Classifiers.VGG16Classifier import VGG16Classifier
from Networks.Encoders.ResNet50Encoder import ResNet50Encoder
from Networks.Encoders.ResNet18Encoder import ResNet18Encoder
from Networks.Discriminators.ModalityAdaptationDiscriminator import ModalityAdaptationDiscriminator
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

'''device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

#svhn = LabeledDataset('Code/StoredDatasets/SVHN', 'svhn_training', transforms.ToTensor())
#mnist = LabeledDataset('Code/StoredDatasets/MNIST', 'mnist_training', transforms.ToTensor())
#usps = LabeledDataset('Code/StoredDatasets/USPS', 'usps_training', transforms.ToTensor())
nyurgb = LabeledDataset('/content/drive/MyDrive/StoredDatasets/NYUD2', 'nyud2_training', transforms.ToTensor())
nyudepth = LabeledDataset('/content/drive/MyDrive/StoredDatasets/NYUD2', 'nyud2_testing', transforms.ToTensor())

source_encoder = ResNet18Encoder().to(device)
target_encoder = ResNet18Encoder().to(device)
classifier = nn.Linear(512, 19).to(device)
discriminator = ModalityAdaptationDiscriminator(512).to(device)
batch_size = 128
num_iterations = 20000
classification_criterion = nn.CrossEntropyLoss()
epochs = 10

classification_lr = 0.001
discriminator_lr = target_encoder_lr = 0.0002

print("Starting the experiment")

experiment = Experiment(nyurgb, nyudepth, source_encoder, target_encoder, classifier, discriminator, epochs, batch_size, classification_criterion,
                        classification_lr, discriminator_lr, target_encoder_lr, num_iterations, device)
experiment.run()'''

'''model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
counter = 0
for p in model.parameters():
    counter += 1
print(counter)'''

'''extractor = LabeledDatasetExtractor(ImageProcessor(), 7291, (28, 28))
extractor.extract('Data/USPS/training')
extractor.save('Code/StoredDatasets/USPS', 'usps_training')

extractor = LabeledDatasetExtractor(ImageProcessor(), 2007, (28, 28))
extractor.extract('Data/USPS/testing')
extractor.save('Code/StoredDatasets/USPS', 'usps_testing')'''

'''extractor = LabeledDatasetExtractor(ImageProcessor(), 2817, (224, 224, 3))
extractor.extract('Data/Office-31/amazon/images')
extractor.save('Code/StoredDatasets/Office-31/Amazon', 'amazon')

extractor = LabeledDatasetExtractor(ImageProcessor(), 498, (224, 224, 3))
extractor.extract('Data/Office-31/dslr/images')
extractor.save('Code/StoredDatasets/Office-31/dslr', 'dslr')

extractor = LabeledDatasetExtractor(ImageProcessor(), 795, (224, 224, 3))
extractor.extract('Data/Office-31/webcam/images')
extractor.save('Code/StoredDatasets/Office-31/Webcam', 'Webcam')'''

dataset = LabeledDataset('Code/StoredDatasets/Office-31/dslr', 'dslr')
print(len(dataset))
while True:
    index = int(input("Enter: "))
    print(dataset.class_dict[dataset[index][1]])
    print(numpy.unique(dataset[index][0]))
    plt.imshow(dataset[index][0])
    plt.show()
