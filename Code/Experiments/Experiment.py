import torch
from Training.SourceTrainer import SourceTrainer
from Training.NetworkTester import NetworkTester
from Training.Adapter import Adapter


class Experiment:
    def __init__(self, source_dataset, target_dataset, source_encoder, target_encoder, classifier, discriminator, num_iterations, batch_size, classification_criterion,
                    classification_optimizer, discriminator_optimizer, target_encoder_optimizer, device):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.source_encoder = source_encoder
        self.target_encoder = target_encoder
        self.classifier = classifier
        self.discriminator = discriminator
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.classification_criterion = classification_criterion
        self.classification_optimizer = classification_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.target_encoder_optimizer = target_encoder_optimizer
        self.device = device

    def run(self):
        source_trainer = SourceTrainer(self.source_encoder, self.classifier, self.source_dataset, self.classification_criterion, self.classification_optimizer, self.num_iterations, self.batch_size, self.device)
        source_trainer.train()
        source_tester = NetworkTester(self.source_encoder, self.classifier, self.source_dataset, self.batch_size, self.device)
        print('Testing the accuracy of source encoder on the source domain...')
        source_tester.test()
        self.target_encoder.load_state_dict(self.source_encoder.state_dict())
        target_tester = NetworkTester(self.target_encoder, self.classifier, self.target_dataset, self.batch_size, self.device)
        print('Testing the accuracy of target encoder on the target domain without adaptation (source only)...')
        target_tester.test()

        adapter = Adapter(self.source_encoder, self.target_encoder, self.discriminator, self.classifier, self.source_dataset, self.target_dataset,
                             self.discriminator_optimizer, self.target_encoder_optimizer, self.num_iterations, self.batch_size, self.device)
        adapter.adapt()
        print('Testing the accuracy of target encoder on the target domain after adaptation...')
        target_tester.test()