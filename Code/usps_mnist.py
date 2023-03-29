import torch
from Datasets.LabeledDataset import LabeledDataset
from torchvision import transforms
from Networks.Encoders.LeNetEncoder import LeNetEncoder
from Networks.Discriminators.DigitsDiscriminator import DigitsDiscriminator
from Experiments.Experiment import Experiment

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

usps = LabeledDataset('Code/StoredDatasets/USPS', 'usps_training', transforms.ToTensor(), sample_size=1800)
mnist = LabeledDataset('Code/StoredDatasets/MNIST', 'mnist_training', transforms.ToTensor(), sample_size=2000)

source_encoder = LeNetEncoder().to(device)
target_encoder = LeNetEncoder().to(device)
classifier = torch.nn.Linear(500, 10).to(device)
discriminator = DigitsDiscriminator().to(device)
classification_epochs = 10
batch_size = 128
adaptation_iterations = 10000
classification_criterion = torch.nn.CrossEntropyLoss()

classification_optimizer = torch.optim.Adam(list(source_encoder.parameters()) + list(classifier.parameters()), lr=0.001)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
target_encoder_optimizer = torch.optim.Adam(target_encoder.parameters(), lr=0.0002, betas=(0.5, 0.999))

print('Starting the USPS --> MNIST experiment.')

experiment = Experiment(usps, mnist, source_encoder, target_encoder, classifier, discriminator, classification_epochs, batch_size, classification_criterion,
                        classification_optimizer, discriminator_optimizer, target_encoder_optimizer, adaptation_iterations, device)
experiment.run()