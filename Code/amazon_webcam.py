import torch
from Datasets.LabeledDataset import LabeledDataset
from torchvision import transforms
from Networks.Encoders.ResNet18Encoder import ResNet18Encoder
from Networks.Discriminators.OfficeDiscriminator import OfficeDiscriminator
from Experiments.Experiment import Experiment

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

amazon = LabeledDataset('Code/StoredDatasets/Office-31/Amazon', 'amazon', transforms.ToTensor())
webcam = LabeledDataset('Code/StoredDatasets/Office-31/Webcam', 'webcam', transforms.ToTensor())

source_encoder = ResNet18Encoder().to(device)
target_encoder = ResNet18Encoder().to(device)
classifier = torch.nn.Linear(512, 19).to(device)
discriminator = OfficeDiscriminator(512).to(device)
classification_epochs = 909
batch_size = 128
adaptation_iterations = 20000
classification_criterion = torch.nn.CrossEntropyLoss()

classification_optimizer = torch.optim.SGD(list(source_encoder.parameters()) + list(classifier.parameters()), lr=0.001)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
target_encoder_optimizer = torch.optim.Adam(target_encoder.parameters(), lr=0.0002, betas=(0.5, 0.999))

print('Starting the Amazon --> Webcam experiment.')

experiment = Experiment(amazon, webcam, source_encoder, target_encoder, classifier, discriminator, classification_epochs, batch_size, classification_criterion,
                        classification_optimizer, discriminator_optimizer, target_encoder_optimizer, adaptation_iterations, device)
experiment.run()