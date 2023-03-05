from Extractors.LabeledDatasetExtractor import LabeledDatasetExtractor
from Processors.ImageProcessor import ImageProcessor
from Datasets.LabeledDataset import LabeledDataset
from torchvision import transforms
from Networks.Encoders.LeNetEncoder import LeNetEncoder
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

b = LabeledDataset('StoredDatasets/MNIST', 'training', transforms.Compose([transforms.ToTensor()]))
print("dataset size:", b.__len__(), b.data.shape, b.labels.shape)

loader = DataLoader(b, 4)

model = LeNetEncoder()

for data, label in loader:
    print(data.shape)
    pred = model(data)
    print(pred, pred.shape)
    input()