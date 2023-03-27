from Extractors.LabeledDatasetExtractor import LabeledDatasetExtractor
from Processors.ImageProcessor import ImageProcessor
from Datasets.LabeledDataset import LabeledDataset
from torchvision import transforms
from Networks.Encoders.LeNetEncoder import LeNetEncoder
from Networks.Discriminators.LeNetDiscriminator import LeNetDiscriminator
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Experiments.Experiment import Experiment
import torch
import torch.nn as nn
import torchvision


mnist = LabeledDataset('Code/StoredDatasets/MNIST', 'training', transforms.Compose([transforms.ToTensor()]), sample_size=2000)
usps = LabeledDataset('Code/StoredDatasets/USPS', 'usps_training', transforms.Compose([transforms.ToTensor()]), sample_size=1800)

source_encoder = LeNetEncoder()
target_encoder = LeNetEncoder()
classifier = nn.Linear(500, 10)
discriminator = LeNetDiscriminator()
batch_size = 128
num_iterations = 10000
classification_criterion = nn.CrossEntropyLoss()

classification_lr = 0.001
discriminator_lr = target_encoder_lr = 0.0002

experiment = Experiment(mnist, usps, source_encoder, target_encoder, classifier, discriminator, batch_size, classification_criterion,
                        classification_lr, discriminator_lr, target_encoder_lr, num_iterations, 'cpu')
experiment.run()