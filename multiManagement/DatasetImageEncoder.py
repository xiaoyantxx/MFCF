import os

import torch
from torchvision.transforms import transforms
from PIL import Image


class DatasetImageEncoder(object):

    @staticmethod
    def images_encoder(train_images, val_images):
        transform_train = transforms.Compose([
            transforms.Resize((448, 448), Image.BILINEAR),
            transforms.RandomHorizontalFlip(),  # solo se train
            transforms.ToTensor(),
        ])

        transform_val = transforms.Compose([
            transforms.Resize((448, 448), Image.BILINEAR),
            transforms.ToTensor(),
        ])
        # train_num_batches = len(train_images) // batch_dataloader  # 76
        train_processed_imgs = []
        '''for i in range(train_num_batches + 1):
            print(i, ' is ok')
            start_idx = i * batch_dataloader
            end_idx = min((i + 1) * batch_dataloader, len(train_images))
            batch_imgs = train_images[start_idx:end_idx]'''
        for path in train_images:
            img = Image.open(path)
            img = img.convert('RGB')
            img = transform_train(img)
            train_processed_imgs.append(img)
        train_i = torch.stack(train_processed_imgs, dim=0)
        # val_num_batches = len(val_images) // batch_dataloader
        val_processed_imgs = []
        '''for i in range(val_num_batches + 1):
            start_idx = i * batch_dataloader
            end_idx = min((i + 1) * batch_dataloader, len(val_images))
            batch_imgs = val_images[start_idx:end_idx]'''
        for path in val_images:
            img = Image.open(path)
            img = img.convert('RGB')
            img = transform_val(img)
            val_processed_imgs.append(img)
        val_i = torch.stack(val_processed_imgs, dim=0)
        return train_i, val_i
