from Extractors.LabeledDatasetExtractor import LabeledDatasetExtractor
from Processors.JPGProcessor import JPGProcessor

a = LabeledDatasetExtractor(None, 1000, (640, 480))
print(a.extract("/home/ali/Downloads/MNIST-JPG-master/MNIST Dataset JPG format/MNIST - JPG - training"))
b = JPGProcessor()
b.process("Ali")