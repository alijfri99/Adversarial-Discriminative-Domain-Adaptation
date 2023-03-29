import torch
from Training.SourceTrainer import SourceTrainer
from Training.NetworkTester import NetworkTester
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy


class Adapter:
    def __init__(self, source_encoder, target_encoder, discriminator, classifier, source_dataset, target_dataset, discriminator_optimizer, target_encoder_optimizer, num_iterations, batch_size, device):
        self.source_encoder = source_encoder
        self.target_encoder = target_encoder
        self.discriminator = discriminator
        self.classifier = classifier
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
        

    def adapt(self):
        for iteration in range(self.num_iterations):
            discriminator_loss = self.train_discriminator()
            target_encoder_loss = self.train_target_encoder()
            
            if (iteration + 1) % 100 == 0:
                print(f'Adaptation: Iteration {iteration + 1}/{self.num_iterations}, Discriminator Loss: {discriminator_loss:.2f}, Target Encoder Loss: {target_encoder_loss:.2f}')
                print('-' * 20)
                target_target = NetworkTester(self.target_encoder, self.classifier, self.target_dataset, self.batch_size, self.device)
                target_target.test()

    def train_discriminator(self):
        self.discriminator_optimizer.zero_grad()

        source_batch = next(iter(self.source_loader))
        source_batch = source_batch[:][0].to(self.device)
        source_batch = source_batch.type(torch.float)
        real_data = self.source_encoder(source_batch).detach()
        discriminator_real_predictions = self.discriminator(real_data)
        label_one = torch.ones((self.batch_size, 1)).to(self.device)
        discriminator_real_loss = self.criterion(discriminator_real_predictions, label_one)
        
        target_batch = next(iter(self.target_loader))
        target_batch = target_batch[:][0].to(self.device)
        target_batch = target_batch.type(torch.float)
        fake_data = self.target_encoder(target_batch).detach()
        discriminator_fake_predictions = self.discriminator(fake_data)
        label_zero = torch.zeros((self.batch_size, 1)).to(self.device)
        discriminator_fake_loss = self.criterion(discriminator_fake_predictions, label_zero)

        discriminator_loss = discriminator_real_loss + discriminator_fake_loss
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        return discriminator_loss.item()

    def train_target_encoder(self):
        self.target_encoder_optimizer.zero_grad()

        target_batch = next(iter(self.target_loader))
        target_batch = target_batch[:][0].to(self.device)
        target_batch = target_batch.type(torch.float)
        target_encoder_predictions = self.target_encoder(target_batch)
        discriminator_predictions = self.discriminator(target_encoder_predictions)
        label_one = torch.ones((self.batch_size, 1)).to(self.device)
        
        target_encoder_loss = self.criterion(discriminator_predictions, label_one)
        target_encoder_loss.backward()
        self.target_encoder_optimizer.step()

        return target_encoder_loss.item()
