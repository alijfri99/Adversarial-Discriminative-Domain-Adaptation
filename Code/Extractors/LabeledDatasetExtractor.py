from Extractors.Extractor import Extractor

class LabeledDatasetExtractor(Extractor):
    def __init__(self, processor, dataset_size, feature_shape):
        super().__init__(processor, dataset_size, feature_shape)

    def extract(self, dataset_root):
        return dataset_root + "53"