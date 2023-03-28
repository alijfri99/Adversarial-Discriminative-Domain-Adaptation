import numpy
from PIL import Image
from Processors.Processor import Processor

class ImageProcessor(Processor):
    def __init__(self, convert_grayscale=False):
        self.target_size = (0, 0)
        self.convert_grayscale = convert_grayscale

    def set_target_size(self, target_size):
        self.target_size = (target_size[0], target_size[1])

    def process(self, path):
        image = Image.open(path)
        image = image.resize(self.target_size)
        image = numpy.array(image, dtype=numpy.float32)
        
        if self.convert_grayscale and len(image.shape) == 3:
            image = numpy.moveaxis(image, -1, 0)
            image = 0.2125 * image[0] + 0.7154 * image[1] + 0.0721 * image[2]

        image = (image - numpy.min(image)) * 255 / (numpy.max(image) - numpy.min(image))
        image = numpy.round(image)
        image = image.astype(numpy.uint8)

        return image