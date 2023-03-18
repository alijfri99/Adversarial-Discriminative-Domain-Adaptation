import copy
from Training.SourceTrainer import SourceTrainer
from Training.NetworkTester import NetworkTester
import torch.nn as nn

class Adapter:
    def __init__(self, source_encoder, source_dataset, target_dataset, criterion, optimizer, num_epochs, batch_size, device):
        self.source_encoder = source_encoder
        self.target_encoder = copy.deepcopy(self.source_encoder)
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.classifier = nn.Linear(500, 10)

    def adapt(self):
        source_trainer = SourceTrainer(self.target_encoder, self.classifier, self.target_dataset, self.criterion, self.optimizer, self.num_epochs, self.batch_size, self.device)
        source_trainer.train()
        tester1 = NetworkTester(self.target_encoder, self.classifier, self)