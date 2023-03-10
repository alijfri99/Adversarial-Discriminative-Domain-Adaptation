import torch
import torch.optim as optim
from torch.utils.data import DataLoader

class SourceTrainer:
    def __init__(self, source_encoder, classifier,  source_dataset, criterion,
                     num_epochs, batch_size, learning_rate, device, shuffle=True):
        self.source_encoder = source_encoder
        self.classifier = classifier
        self.source_dataset = source_dataset
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.shuffle = shuffle
        
    def train_source(self):
        train_loader = DataLoader(self.source_dataset, self.batch_size, self.shuffle)
        
        optimizer = optim.Adam(list(self.source_encoder.parameters()) + list(self.classifier.parameters()), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
             for index, (data, labels) in enumerate(train_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()

                features = self.source_encoder(data)
                predictions = self.classifier(features)
                loss = self.criterion(predictions, labels)

                loss.backward()
                optimizer.step()
