import numpy
from PIL import Image
from Processors.Processor import Processor

class JPGProcessor(Processor):
    def process(self, path):
        image = Image.open(path)
        image = numpy.asarray(image)
        return image