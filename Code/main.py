from Extractors.LabeledDatasetExtractor import LabeledDatasetExtractor
from Processors.ImageProcessor import ImageProcessor
from Datasets.LabeledDataset import LabeledDataset
from torchvision import transforms
from Networks.Encoders.LeNetEncoder import LeNetEncoder
from Networks.Discriminators.LeNetDiscriminator import LeNetDiscriminator
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Training.SourceTrainer import SourceTrainer
from Training.NetworkTester import NetworkTester
from Training.Adapter import Adapter
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import numpy

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

classification_criterion = nn.CrossEntropyLoss()

classification_optimizer = torch.optim.Adam(list(source_encoder.parameters()) + list(classifier.parameters()), lr=0.001)
source_trainer = SourceTrainer(source_encoder, classifier, svhn, classification_criterion, classification_optimizer, 10, batch_size, 'cpu')
source_trainer.train()
source_tester = NetworkTester(source_encoder, classifier, svhn, batch_size, 'cpu')
source_tester.test()

target_encoder.load_state_dict(source_encoder.state_dict())

target_tester = NetworkTester(target_encoder, classifier, mnist, batch_size, 'cpu')
target_tester.test()

discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
target_encoder_optimizer = torch.optim.Adam(target_encoder.parameters(), lr=0.0002, betas=(0.5, 0.999))

adapter = Adapter(source_encoder, target_encoder, discriminator, svhn, mnist, discriminator_optimizer, target_encoder_optimizer, 10000, 128, 'cpu')
adapter.adapt()
target_tester.test()
