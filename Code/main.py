from Extractors.LabeledDatasetExtractor import LabeledDatasetExtractor
from Processors.ImageProcessor import ImageProcessor
from Datasets.LabeledDataset import LabeledDataset
from torchvision import transforms
import matplotlib.pyplot as plt

a = LabeledDataset('StoredDatasets/MNIST', 'training', transforms.Compose([transforms.ToTensor(), transforms.Resize((1024, 1024))]))
print("dataset size:", a.__len__(), a.data.shape, a.labels.shape)
while True:
    inp = int(input("Enter an index: "))
    data, label = a.__getitem__(inp)
    l = a.class_dict[a.labels[inp]]
    plt.imshow(data.numpy()[0], cmap='gray')
    print(l)
    print(data.shape)
    plt.show()