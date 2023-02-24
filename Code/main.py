from Extractors.LabeledDatasetExtractor import LabeledDatasetExtractor
from Processors.ImageProcessor import ImageProcessor
import matplotlib.pyplot as plt

a = LabeledDatasetExtractor(ImageProcessor(), 60000, (28, 28))
a.extract("/home/alijfri99/Projects/Unsupervised-Domain-Adaptation/Data/MNIST Dataset JPG format/MNIST - JPG - testing")
while True:
    inp = int(input("Enter an index: "))
    plt.imshow(a.data[inp], cmap='gray')
    print(a.labels[inp])
    plt.show()