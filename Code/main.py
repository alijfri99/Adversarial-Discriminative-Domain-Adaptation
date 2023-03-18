from Extractors.LabeledDatasetExtractor import LabeledDatasetExtractor
from Processors.ImageProcessor import ImageProcessor
from Datasets.LabeledDataset import LabeledDataset
from torchvision import transforms
from Networks.Encoders.LeNetEncoder import LeNetEncoder
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Training.SourceTrainer import SourceTrainer
from Training.NetworkTester import NetworkTester
from Training.Adapter import Adapter
import torch
import torch.nn as nn
import torchvision


mnist = LabeledDataset('Code/StoredDatasets/MNIST', 'mnist_training', transforms.Compose([transforms.ToTensor()]), sample_size=2000)
usps = LabeledDataset('Code/StoredDatasets/USPS', 'usps_training', transforms.Compose([transforms.ToTensor()]), sample_size=1800)
source_encoder = LeNetEncoder()
target_encoder = LeNetEncoder()
classifier = nn.Linear(500, 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(target_encoder.parameters()) + list(classifier.parameters()), lr = 0.001)
adaper = Adapter(classifier, source_encoder, target_encoder, mnist, usps, criterion, optimizer, 5, 4, 'cpu')
adaper.adapt()

'''
#labeledDataset = LabeledDataset('Code/StoredDatasets/MNIST', 'training', transforms.Compose([transforms.ToTensor()]))
labeledDataset = LabeledDataset('Code/StoredDatasets/USPS', 'usps_training', transforms.Compose([transforms.ToTensor()]), sample_size=1800)
print("dataset size:", labeledDataset.__len__(), labeledDataset.data.shape, labeledDataset.labels.shape)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

num_epoch = 2
batch_size = 100
learning_rate = 0.001

loader = DataLoader(labeledDataset, 4)
encoder = LeNetEncoder()

classifier = nn.Linear(500, 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr = learning_rate)

sourceTrainer = SourceTrainer(encoder, classifier, labeledDataset, criterion, optimizer, num_epoch, batch_size, device, True)
sourceTrainer.train()

netTester = NetworkTester(encoder, classifier, labeledDataset, batch_size, device)
netTester.test()
'''


'''
# USPS Dataset Extractor (28, 28)
target_size = (28, 28)
training_dataset_size = 60000 
testing_dataset_size = 10000

imageProcessor = ImageProcessor()
training_dataset_extractor = LabeledDatasetExtractor(imageProcessor, training_dataset_size, target_size)
training_dataset_extractor.extract('Data/MNIST/training')
training_dataset_extractor.save('Code/StoredDatasets/MNIST', 'mnist_training')

testing_dataset_extractor = LabeledDatasetExtractor(imageProcessor, testing_dataset_size, target_size)
testing_dataset_extractor.extract('Data/MNIST/testing')
testing_dataset_extractor.save('Code/StoredDatasets/MNIST', 'mnist_testing')
'''

