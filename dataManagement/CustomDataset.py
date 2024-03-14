from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data):
        self.labels = data.labels
        self.images = data.images
        self.texts = data.texts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]
        text = self.texts[idx]

        sample = [label, image, text]
        return sample


