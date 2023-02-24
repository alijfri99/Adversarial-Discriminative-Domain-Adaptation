from Extractors.LabeledDatasetExtractor import LabeledDatasetExtractor
from Processors.ImageProcessor import ImageProcessor
from Datasets.LabeledDataset import LabeledDataset
from torchvision import transforms
import matplotlib.pyplot as plt

a = LabeledDataset("StoredDatasets/MNIST", "training", transforms.ToTensor())
print("DONE!", len(a))
while True:
    inp = int(input("Enter an index: "))
    b = a.__getitem__(inp)
    print(b, type(b))
    input()
    plt.imshow(a.data[inp], cmap='gray')
    print(a.labels[inp])
    print(a.class_dict[a.labels[inp]])
    plt.show()