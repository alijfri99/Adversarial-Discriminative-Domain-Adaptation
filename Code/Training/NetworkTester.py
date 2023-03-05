import torch
import torch.optim as optim
from torch.utils.data import DataLoader

class NetworkTester:
    def __init__(self, target_encoder, classifier,  target_dataset):
        self.target_encoder = target_encoder
        self.classifier = classifier
        self.target_dataset = target_dataset


    def test_network(self, batch_size, device):
        train_loader = DataLoader(self.target_dataset, batch_size, False)
        
        with torch.no_grad():
            n_correct = 0
            n_samples = 0

            n_class_correct = [0 for i in range(10)]
            n_class_samples = [0 for i in range(10)]

            for data, labels in test_loader:
                data = data.to(device)
                labels = labels.to(device)

                features = self.target_encoder(data)
                predictions = self.classifier(features)

                n_samples += labels.size(0)
                n_correct += (predictions == labels).sum().item()

                for i in range(batch_size):
                    label = labels[i]
                    pred = predictions[i]
                    if (label == pred):
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1

            Accuracy = 100.0 * n_correct/n_samples
            print(f'Accuracy of the Network = {Accuracy} %')

            for i in range(10):
                Accuracy = 100.0 *  n_class_correct[i]/ n_class_samples[i]
                print(f'Accuracy of = {classes[i]} : {Accuracy} %')