import torch
import torch.optim as optim
from torch.utils.data import DataLoader

class NetworkTester:
    def __init__(self, encoder, classifier,  target_dataset, batch_size, device):
        self.encoder = encoder
        self.classifier = classifier
        self.target_dataset = target_dataset
        self.batch_size = batch_size
        self.device = device


    def test(self):
        test_loader = DataLoader(self.target_dataset, self.batch_size, False)
        
        with torch.no_grad():
            n_correct = 0
            n_samples = 0

            n_class_correct = [0 for i in range(10)]
            n_class_samples = [0 for i in range(10)]

            for data, labels in test_loader:
                data = data.type(torch.float)
                labels = labels.type(torch.LongTensor)
                data = data.to(self.device)
                labels = labels.to(self.device)

                features = self.encoder(data)
                predictions = self.classifier(features)
                predictions = torch.argmax(predictions, dim=1)
                n_samples += labels.size(0)
                n_correct += (predictions == labels).sum().item()

                for i in range(len(data)):
                    label = labels[i]
                    pred = predictions[i]
                    if (label == pred):
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1

            Accuracy = 100.0 * n_correct/n_samples
            print(f'Accuracy of the Network: {Accuracy:.2f} %')

            for i in range(10):
                Accuracy = 100.0 *  n_class_correct[i]/ n_class_samples[i]
                print(f'Accuracy of class {i}: {Accuracy:.2f} %')