from Extractors.LabeledDatasetExtractor import LabeledDatasetExtractor
from Processors.ImageProcessor import ImageProcessor
from Datasets.LabeledDataset import LabeledDataset
from torchvision import transforms
import matplotlib.pyplot as plt

a = LabeledDataset("StoredDatasets/MNIST", "training", transforms.Compose([transforms.ToTensor(), transforms.RandomRotation(90)]))
print("DONE!", len(a))
while True:
    inp = int(input("Enter an index: "))
    b = a.__getitem__(inp)
    print(b, type(b))
    plt.imshow(a.__getitem__(inp)[0].numpy()[0], cmap='gray')
    print(a.labels[inp])
    print(a.class_dict[a.labels[inp]])
    plt.show()