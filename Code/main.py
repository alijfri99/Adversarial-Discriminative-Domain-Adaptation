from Extractors.LabeledDatasetExtractor import LabeledDatasetExtractor
from Processors.ImageProcessor import ImageProcessor
from Datasets.LabeledDataset import LabeledDataset
from torchvision import transforms
from Networks.Encoders.LeNetEncoder import LeNetEncoder
import matplotlib.pyplot as plt

b = LabeledDataset('StoredDatasets/MNIST', 'training', transforms.Compose([transforms.ToTensor()]))
print("dataset size:", b.__len__(), b.data.shape, b.labels.shape)

while True:
    inp = int(input("Enter an index: "))
    data, label = b.__getitem__(inp)
    print(data.dtype)
    model = LeNetEncoder()
    temp = model(data)
    print(temp, temp.shape)