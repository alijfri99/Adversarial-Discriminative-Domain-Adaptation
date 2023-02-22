import numpy

class Extractor:
    def __init__(self, processor, dataset_size, feature_shape):
        self.processor = processor
        self.dataset_size = dataset_size
        self.feature_shape = feature_shape
        self.data = numpy.zeros((dataset_size, ) + feature_shape)
        self.labels = numpy.zeros(dataset_size)

    def extract(dataset_root):
        raise NotImplementedError
    