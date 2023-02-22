from Extractors.LabeledDatasetExtractor import LabeledDatasetExtractor
from Processors.ImageProcessor import ImageProcessor
import matplotlib.pyplot as plt

a = LabeledDatasetExtractor(ImageProcessor(), 60000, (28, 28))
a.extract("/home/ali/Downloads/MNIST-JPG-master/MNIST Dataset JPG format/MNIST - JPG - training")
while True:
    inp = int(input("Enter an index: "))
    plt.imshow(a.data[inp], cmap='gray')
    print(a.labels[inp])
    plt.show()