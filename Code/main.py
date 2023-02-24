from Extractors.LabeledDatasetExtractor import LabeledDatasetExtractor
from Processors.ImageProcessor import ImageProcessor
from Datasets.LabeledDataset import LabeledDataset
import matplotlib.pyplot as plt

a = LabeledDataset("StoredDatasets/MNIST", "training")
print("DONE!")
while True:
    inp = int(input("Enter an index: "))
    plt.imshow(a.data[inp], cmap='gray')
    print(a.labels[inp])
    print(a.class_dict[inp])
    plt.show()