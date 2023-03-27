import numpy
from PIL import Image
from Processors.Processor import Processor

class ImageProcessor(Processor):
    def __init__(self):
        self.target_size = (0, 0)

    def set_target_size(self, target_size):
        self.target_size = (target_size[0], target_size[1])

    def process(self, path):
        image = Image.open(path)
        image = image.resize(self.target_size)
        image = numpy.array(image, dtype=numpy.float32)
        
        if len(image.shape) == 3:
            image = numpy.moveaxis(image, -1, 0)
            image = 0.2125 * image[0] + 0.7154 * image[1] + 0.0721 * image[2]

        return image