import torch
from Training.SourceTrainer import SourceTrainer
from Training.NetworkTester import NetworkTester
from torch.utils.data import DataLoader
import torch.nn as nn


class Adapter:
    def __init__(self, source_encoder, target_encoder, discriminator, source_dataset, target_dataset, discriminator_optimizer, target_encoder_optimizer, num_iterations, batch_size, device):
        self.source_encoder = source_encoder
        self.target_encoder = target_encoder
        self.discriminator = discriminator
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.discriminator_optimizer = discriminator_optimizer
        self.target_encoder_optimizer = target_encoder_optimizer
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.device = device

        self.source_loader = DataLoader(self.source_dataset, self.batch_size, True)
        self.target_loader = DataLoader(self.target_dataset, self.batch_size, True)
        self.criterion = nn.BCELoss()
        self.label_zero = torch.zeros((self.batch_size, 1), dtype=torch.float32)
        self.label_one = torch.ones((self.batch_size, 1), dtype=torch.float32)

    def adapt(self):
        pass

    def train_discriminator(self):
        self.discriminator_optimizer.zero_grad()

        source_batch = next(iter(self.source_loader))
        source_batch = source_batch[:][0]
        real_data = self.source_encoder(source_batch).detach()
        discriminator_real_predictions = self.discriminator(real_data)
        discriminator_real_loss = self.criterion(discriminator_real_predictions, self.label_one)
        
        target_batch = next(iter(self.target_loader))
        target_batch = target_batch[:][0]
        fake_data = self.target_encoder(target_batch).detach()
        discriminator_fake_predictions = self.discriminator(fake_data)
        discriminator_fake_loss = self.criterion(discriminator_fake_predictions, self.label_zero)

        discriminator_loss = 0.5 * (discriminator_real_loss + discriminator_fake_loss)
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

    def train_target_encoder(self):
        self.target_encoder_optimizer.zero_grad()

        target_batch = next(iter(self.target_loader))
        target_batch = target_batch[:][0]
        target_encoder_predictions = self.target_encoder(target_batch)
        discriminator_predictions = self.discriminator(target_encoder_predictions)
        
        target_encoder_loss = self.criterion(discriminator_predictions, self.label_one)
        target_encoder_loss.backward()
        self.target_encoder_optimizer.step()
