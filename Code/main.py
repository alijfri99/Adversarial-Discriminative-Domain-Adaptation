from Extractors.LabeledDatasetExtractor import LabeledDatasetExtractor

a = LabeledDatasetExtractor(None, 1000, (640, 480))
print(a.extract("Ali"))