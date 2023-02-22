import os
from Extractors.Extractor import Extractor

class LabeledDatasetExtractor(Extractor):
    def __init__(self, processor, dataset_size, feature_shape):
        super().__init__(processor, dataset_size, feature_shape)

    def extract(self, dataset_root):
        self.class_dict.clear()
        class_names = os.listdir(dataset_root)
        class_names.sort()
        data_array_index = 0

        for class_index, class_name in enumerate(class_names):
            self.class_dict[class_index] = class_name
            datapoints_path = os.path.join(dataset_root, class_name)
            datapoints = os.listdir(datapoints_path)

            for datapoint in datapoints:
                datapoint_path = os.path.join(datapoints_path, datapoint)
                data_array = self.processor.process(datapoint_path)
                self.data[data_array_index] = data_array
                self.labels[data_array_index] = class_index
                data_array_index += 1
            
            print(f'Class {class_name} done.')