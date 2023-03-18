import copy
from Training.SourceTrainer import SourceTrainer
from Training.NetworkTester import NetworkTester
import torch.nn as nn

class Adapter:
    def __init__(self, classifier, source_encoder, target_encoder, source_dataset, target_dataset, criterion, optimizer, num_epochs, batch_size, device):
        self.source_encoder = source_encoder
        self.target_encoder = target_encoder
        self.target_encoder.load_state_dict(source_encoder.state_dict())
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.classifier = classifier

    def adapt(self):
        source_trainer = SourceTrainer(self.target_encoder, self.classifier, self.target_dataset, self.criterion, self.optimizer, self.num_epochs, self.batch_size, self.device)
        source_trainer.train()
        tester1 = NetworkTester(self.target_encoder, self.classifier, self.target_dataset, 4, 'cpu')
        tester1.test()
        tester2 = NetworkTester(self.target_encoder, self.classifier, self.source_dataset, 4, 'cpu')
        tester2.test()
        tester3 = NetworkTester(self.source_encoder, self.classifier, self.target_dataset, 4, 'cpu')
        tester3.test()
        tester4 = NetworkTester(self.source_encoder, self.classifier, self.source_dataset, 4, 'cpu')
        tester4.test()