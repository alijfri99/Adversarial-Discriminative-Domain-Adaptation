import torch
import torch.optim as optim
from torch.utils.data import DataLoader

class SourceTrainer:
    def __init__(self, source_encoder, classifier,  source_dataset):
        self.source_encoder = source_encoder
        self.classifier = classifier
        self.source_dataset = source_dataset

        
    def train_source(self, num_epochs, batch_size, learning_rate, device, shuffle=True):
        train_loader = DataLoader(self.source_dataset, batch_size, shuffle)
        criterion = nn.CrossEntropyLoss() #should we also get this as parameter?
        optimizer = optim.Adam(list(self.source_encoder.parameters()) + list(self.classifier.parameters()) , lr=learning_rate)

        for epoch in range(num_epochs):
             for index, (data, labels) in enumerate(train_loader):
                data = data.to(device)
                labels = labels.to(device)

                features = self.source_encoder(data)
                predictions = self.classifier(features)
                loss = criterion(predictions, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return source_encoder, classifier