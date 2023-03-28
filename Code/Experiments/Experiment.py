import torch
from Training.SourceTrainer import SourceTrainer
from Training.NetworkTester import NetworkTester
from Training.Adapter import Adapter


class Experiment:
    def __init__(self, source_dataset, target_dataset, source_encoder, target_encoder, classifier, discriminator, epochs, batch_size, classification_criterion,
                    classification_lr,  discriminator_lr, target_encoder_lr, num_iterations, device):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.source_encoder = source_encoder
        self.target_encoder = target_encoder
        self.classifier = classifier
        self.discriminator = discriminator
        self.epochs = epochs
        self.batch_size = batch_size
        self.classification_criterion = classification_criterion
        self.classification_lr = classification_lr
        self.discriminator_lr = discriminator_lr
        self.target_encoder_lr = target_encoder_lr
        self.num_iterations = num_iterations
        self.device = device


    def run(self):
        classification_optimizer = torch.optim.Adam(list(self.source_encoder.parameters()) + list(self.classifier.parameters()), lr=self.classification_lr)
        source_trainer = SourceTrainer(self.source_encoder, self.classifier, self.source_dataset, self.classification_criterion, classification_optimizer, self.epochs, self.batch_size, self.device)
        source_trainer.train()
        source_tester = NetworkTester(self.source_encoder, self.classifier, self.source_dataset, self.batch_size, self.device)
        source_tester.test()
        self.target_encoder.load_state_dict(self.source_encoder.state_dict())

        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.discriminator_lr, betas=(0.5, 0.999))
        target_encoder_optimizer = torch.optim.Adam(self.target_encoder.parameters(), lr=self.target_encoder_lr, betas=(0.5, 0.999))

        adapter = Adapter(self.source_encoder, self.target_encoder, self.discriminator, self.classifier, self.source_dataset, self.target_dataset,
                             discriminator_optimizer, target_encoder_optimizer, self.num_iterations, self.batch_size, self.device)
        adapter.adapt()
        target_tester = NetworkTester(self.target_encoder, self.classifier, self.target_dataset, self.batch_size, self.device)
        target_tester.test()