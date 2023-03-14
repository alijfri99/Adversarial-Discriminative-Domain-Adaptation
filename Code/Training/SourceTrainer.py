import torch
import torch.optim as optim
from torch.utils.data import DataLoader

class SourceTrainer:
    def __init__(self, source_encoder, classifier,  source_dataset, criterion, optimizer,
                     num_epochs, batch_size, device, shuffle=True):
        self.source_encoder = source_encoder
        self.classifier = classifier
        self.source_dataset = source_dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        
    def train(self):
        train_loader = DataLoader(self.source_dataset, self.batch_size, self.shuffle)

        for epoch in range(self.num_epochs):
             for index, (data, labels) in enumerate(train_loader):
                labels = labels.type(torch.LongTensor)
                data = data.type(torch.float)

                data = data.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()

                features = self.source_encoder(data)
                predictions = self.classifier(features)
                loss = self.criterion(predictions, labels)

                loss.backward()
                self.optimizer.step()

        print("Finished Training")