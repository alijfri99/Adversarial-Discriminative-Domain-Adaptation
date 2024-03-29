import numpy

class Extractor:
    def __init__(self, processor, dataset_size, feature_shape):
        self.processor = processor
        self.dataset_size = dataset_size
        self.feature_shape = feature_shape
        self.data = numpy.zeros((dataset_size, ) + feature_shape, dtype=numpy.uint8)
        self.labels = numpy.zeros(dataset_size, dtype=numpy.uint8)
        self.class_dict = dict()

    def extract(self, dataset_root):
        raise NotImplementedError
    
    def save(self, dataset_output_path, output_file_name):
        raise NotImplementedError
    