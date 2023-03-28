from torchvision import models
from Networks.Encoders.Identity import Identity

class ResNet18Encoder:
    def __new__(cls):
        encoder = models.resnet18(models.ResNet18_Weights.IMAGENET1K_V1)
        for parameter in encoder.layer4.parameters():
            parameter.requires_grad = False
        encoder.fc = Identity()
        return encoder
    