from Datasets.LabeledDataset import LabeledDataset
from torchvision import transforms
from Networks.Encoders.LeNetEncoder import LeNetEncoder
from Networks.Discriminators.DigitsDiscriminator import DigitsDiscriminator
from Experiments.Experiment import Experiment
import torch
import torch.nn as nn

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

mnist = LabeledDataset('Code/StoredDatasets/MNIST', 'mnist_training', transforms.ToTensor(), sample_size=2000)
usps = LabeledDataset('Code/StoredDatasets/USPS', 'usps_training', transforms.ToTensor(), sample_size=1800)

source_encoder = LeNetEncoder().to(device)
target_encoder = LeNetEncoder().to(device)
classifier = nn.Linear(500, 10).to(device)
discriminator = DigitsDiscriminator().to(device)
batch_size = 128
num_iterations = 10000
classification_criterion = nn.CrossEntropyLoss()
epochs = 10

classification_lr = 0.001
adaptation_lr = 0.0002

print('Starting the MNIST --> USPS experiment.')

experiment = Experiment(mnist, usps, source_encoder, target_encoder, classifier, discriminator, epochs, batch_size, classification_criterion,
                        classification_lr, adaptation_lr, num_iterations, device)
experiment.run()