import os
import numpy
import pickle
from Extractors.Extractor import Extractor

class LabeledDatasetExtractor(Extractor):
    def __init__(self, processor, dataset_size, feature_shape):
        super().__init__(processor, dataset_size, feature_shape)

    def extract(self, dataset_root):
        self.class_dict.clear()
        class_names = os.listdir(dataset_root)
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

    def save(self, dataset_output_path, output_file_name):
        dataset_output_path = os.path.abspath(dataset_output_path)
        dataset_store_location = os.path.join(dataset_output_path, output_file_name) + ".npz"
        class_dict_store_location = os.path.join(dataset_output_path, output_file_name) + "_class_dict.pkl"

        numpy.savez_compressed(dataset_store_location, data=self.data, labels=self.labels)

        with open(class_dict_store_location, 'wb') as output_file:
            pickle.dump(self.class_dict, output_file)

        print("Saved the dataset and the class dictionary.")