from dataManagement.DatasetSplit import DatasetSplit


class DatasetLoader(object):

    def __init__(self):
        self.train = DatasetSplit()
        self.val = DatasetSplit()

    def load_data(self, train_path, val_path, delimiter):
        print('Loading data...')
        self.train.load_data(train_path, delimiter)
        train_labels = self.train.get_labels()
        train_images = self.train.get_images()
        train_texts = self.train.get_texts()

        self.val.load_data(val_path, delimiter)
        val_labels = self.val.get_labels()
        val_images = self.val.get_images()
        val_texts = self.val.get_texts()

        print('Train/val split: {:d}/{:d}'.format(len(train_texts), len(val_texts)))

        self.set_train_data(train_labels, train_images, train_texts)
        self.set_val_data(val_labels, val_images, val_texts)

    def set_train_data(self, train_labels, train_images, train_texts):
        self.train.set_labels(train_labels)
        self.train.set_images(train_images)
        self.train.set_texts(train_texts)

    def set_val_data(self, val_labels, val_images, val_texts):
        self.val.set_labels(val_labels)
        self.val.set_images(val_images)
        self.val.set_texts(val_texts)

    def get_train_data(self):
        return self.train

    def get_val_data(self):
        return self.val
