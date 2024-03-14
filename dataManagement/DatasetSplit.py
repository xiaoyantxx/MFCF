class DatasetSplit(object):

    def __init__(self):
        self.labels = []
        self.images = []
        self.texts = []

    def load_data(self, data_file, delimiter):
        with open(data_file) as tr:
            for line in tr.readlines():
                line = line.replace('\n', '')
                line = line.split(delimiter)
                self.labels.append(line[0])
                self.images.append(line[1])
                self.texts.append(line[2])

    def get_labels(self):
        return self.labels

    def get_images(self):
        return self.images

    def get_texts(self):
        return self.texts

    def set_labels(self, labels):
        self.labels = labels

    def set_images(self, images):
        self.images = images

    def set_texts(self, texts):
        self.texts = texts
