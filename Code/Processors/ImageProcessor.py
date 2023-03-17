import numpy
from PIL import Image
from Processors.Processor import Processor

class ImageProcessor(Processor):
    def __init__(self):
        self.target_size = (0, 0)

    def set_target_size(self, target_size):
        if len(target_size) > 2:
            if (target_size[-2] > 0  and target_size[-1] > 0):
                self.target_size = (target_size[-2], target_size[-1])
        else:
            if (target_size[0] > 0  and target_size[1] > 0):
                self.target_size = target_size

    def process(self, path):
        image = Image.open(path)
        image = image.resize(self.target_size)
        image = numpy.array(image, dtype=numpy.float32)
        return image