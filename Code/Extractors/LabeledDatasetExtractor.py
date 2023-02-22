import numpy
from Extractors.Extractor import Extractor

class LabeledDatasetExtractor(Extractor):
    def __init__(self, processor, dataset_size, feature_shape):
        super(LabeledDatasetExtractor, self).__init__(processor, dataset_size, feature_shape)