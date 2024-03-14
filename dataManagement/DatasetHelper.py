import torch

from multiManagement.DatasetImageEncoder import DatasetImageEncoder
from multiManagement.TextTokenizer import TextTokenizer
from multiManagement.DatasetLabelEncoder import DatasetLabelEncoder


class DatasetHelper:

    def __init__(self, num_words_to_keep):
        self.label_encoder = DatasetLabelEncoder()
        self.image_encoder = DatasetImageEncoder()
        self.tokenizer = TextTokenizer(num_words_to_keep)

    def preprocess_labels(self, train_data, val_data):
        self.train_one_hot_encoder(train_data.get_labels())

        train_y = self.labels_to_one_hot(train_data.get_labels())
        val_y = self.labels_to_one_hot(val_data.get_labels())
        return train_y, val_y

    def preprocess_images(self, train_images, val_images):
        return self.images_encoder(train_images, val_images)

    def preprocess_texts(self, train_texts, val_texts, num_words_x_doc):
        self.train_tokenizer(train_texts)

        train_t = self.texts_to_indices(train_texts)
        for i in range(len(train_t)):
            if len(train_t[i]) > num_words_x_doc:
                train_t[i] = train_t[i][:num_words_x_doc]
            elif len(train_t[i]) < num_words_x_doc:
                train_t[i].extend([0] * (num_words_x_doc - len(train_t[i])))
        train_t = torch.tensor(train_t)

        val_t = self.texts_to_indices(val_texts)
        for i in range(len(val_t)):
            if len(val_t[i]) > num_words_x_doc:
                val_t[i] = val_t[i][:num_words_x_doc]
            elif len(val_t[i]) < num_words_x_doc:
                val_t[i].extend([0] * (num_words_x_doc - len(val_t[i])))
        val_t = torch.tensor(val_t)
        return train_t, val_t

    def train_one_hot_encoder(self, train_labels):
        return self.label_encoder.train_one_hot_encoder(train_labels)

    def labels_to_one_hot(self, labels):
        return self.label_encoder.encode_to_one_hot(labels)

    def images_encoder(self, train_images, val_images):
        return self.image_encoder.images_encoder(train_images, val_images)

    def train_tokenizer(self, train_texts):
        self.tokenizer.train_tokenizer(train_texts)

    def texts_to_indices(self, texts):
        return self.tokenizer.convert_to_indices(texts)


