from Extractors.LabeledDatasetExtractor import LabeledDatasetExtractor

a = LabeledDatasetExtractor(None, 1000, (640, 480))
print(a.extract("/home/ali/Downloads/MNIST-JPG-master/MNIST Dataset JPG format/MNIST - JPG - training"))