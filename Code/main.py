from Extractors.LabeledDatasetExtractor import LabeledDatasetExtractor
from Processors.ImageProcessor import ImageProcessor
from Datasets.LabeledDataset import LabeledDataset
from torchvision import transforms
import matplotlib.pyplot as plt

a = LabeledDatasetExtractor(ImageProcessor(), 10000, (28, 28))
a.extract('/home/ali/Projects/Unsupervised-Domain-Adaptation/Data/MNIST Dataset JPG format/MNIST - JPG - testing')
while True:
    inp = int(input("Enter an index: "))
    b = a.data[inp]
    l = a.class_dict[a.labels[inp]]
    plt.imshow(b, cmap='gray')
    print(l)
    plt.show()