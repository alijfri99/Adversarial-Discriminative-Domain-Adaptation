from torchvision import models
from Networks.Encoders.Identity import Identity

class VGG16Encoder:
    def __new__(cls):
        encoder = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        encoder.classifier = Identity()
        return encoder