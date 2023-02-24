import numpy
import pickle
import os
from torch.utils.data import Dataset

class LabeledDataset(Dataset):
    def __init__(self, dataset_path, dataset_name, transform=None):
        dataset_store_location = os.path.join(dataset_path, dataset_name) + ".npz"
        class_dict_store_location = os.path.join(dataset_path, dataset_name) + "_class_dict.pkl"

        raw_dataset = numpy.load(dataset_store_location)
        self.data = raw_dataset['data']
        self.labels = raw_dataset['labels']

        with open(class_dict_store_location, 'rb') as input_file:
            self.class_dict = pickle.load(input_file)

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        item_data = self.data[index]
        item_label = self.labels[index]
        
        if self.transform:
            item_data = self.transform(item_data)
        
        return item_data, item_label