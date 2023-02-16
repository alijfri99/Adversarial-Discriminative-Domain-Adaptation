class LabeledDatasetExtractor:
    def __init__(self, processor, dataset_size):
        self.processor = processor
        self.dataset_size = dataset_size