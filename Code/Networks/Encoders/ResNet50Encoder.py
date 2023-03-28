from torchvision import models
from Networks.Encoders.Identity import Identity

class ResNet50Encoder:
    def __new__(cls):
        encoder = models.resnet50(models.ResNet50_Weights.IMAGENET1K_V1)
        for parameter in encoder.layer4.parameters():
            parameter.requires_grad = False
        encoder.fc = Identity()
        return encoder
