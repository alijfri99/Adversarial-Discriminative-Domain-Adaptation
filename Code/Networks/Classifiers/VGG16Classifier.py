import torch.nn as nn
from torchvision import models

class VGG16Classifier:
    def __new__(cls):
        classifier = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        classifier = classifier.classifier
        classifier[6] = nn.Linear(4096, 19)
        return classifier