from Extractors.LabeledDatasetExtractor import LabeledDatasetExtractor
from Processors.ImageProcessor import ImageProcessor
from Datasets.LabeledDataset import LabeledDataset
from torchvision import transforms
from Networks.Encoders.LeNetEncoder import LeNetEncoder
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Training.SourceTrainer import SourceTrainer
from Training.NetworkTester import NetworkTester
import torch
import torch.nn as nn
import torchvision


labeledDataset = LabeledDataset('Code/StoredDatasets/MNIST', 'training', transforms.Compose([transforms.ToTensor()]))
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

