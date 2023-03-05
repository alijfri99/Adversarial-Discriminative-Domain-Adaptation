import numpy
from PIL import Image
from Processors.Processor import Processor

class ImageProcessor(Processor):
    def process(self, path):
        image = Image.open(path)
        image = numpy.array(image, dtype=numpy.float32)
        return image