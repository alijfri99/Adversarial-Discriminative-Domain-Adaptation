import torch
from Datasets.LabeledDataset import LabeledDataset
from torchvision import transforms
from Networks.Encoders.ResNet18Encoder import ResNet18Encoder
from Networks.Discriminators.ModalityAdaptationDiscriminator import ModalityAdaptationDiscriminator
from Experiments.Experiment import Experiment

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

nyurgb = LabeledDataset('Code/StoredDatasets/NYUD2', 'nyud2_training', transforms.ToTensor())
nyudepth = LabeledDataset('Code/StoredDatasets/NYUD2', 'nyud2_testing', transforms.ToTensor())

source_encoder = ResNet18Encoder().to(device)
target_encoder = ResNet18Encoder().to(device)
classifier = torch.nn.Linear(512, 19).to(device)
discriminator = ModalityAdaptationDiscriminator(512).to(device)
num_iterations = 20000
batch_size = 128
classification_criterion = torch.nn.CrossEntropyLoss()

classification_optimizer = torch.optim.Adam(list(source_encoder.parameters()) + list(classifier.parameters()), lr=0.001)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
target_encoder_optimizer = torch.optim.Adam(target_encoder.parameters(), lr=0.0002, betas=(0.5, 0.999))

print('Starting the NYUD2 RGB --> NYUD2 Depth experiment.')

experiment = Experiment(nyurgb, nyudepth, source_encoder, target_encoder, classifier, discriminator, num_iterations, batch_size, classification_criterion,
                        classification_optimizer, discriminator_optimizer, target_encoder_optimizer, device)
experiment.run()