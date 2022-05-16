import torch
import numpy as np
import torchvision.transforms as T


from os import listdir
from torch.utils.data import Dataset


class LiverDEC(Dataset):
    def __init__(self, train, no_aug):

        self.root_path = 'set_path_to_data'
        self.no_aug = no_aug

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if no_aug:
            augmentation = [
                T.ToPILImage(),
                T.Resize(224),
                T.ToTensor(),
            ]
        else:
            augmentation = [
                T.ToPILImage(),
                T.RandomResizedCrop(224, scale=(0.2, 1.)),
                T.RandomApply([
                    T.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
                ], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]

        if train:
            self.img_path = ''.join([self.root_path, 'set_path'])
            self.label_path = ''.join([self.root_path, 'set_path'])
        else:
            self.img_path = ''.join([self.root_path, 'set_path'])
            self.label_path = ''.join([self.root_path, 'set_path'])


        self.img_transform = T.Compose(augmentation)
        self.fnames = listdir(self.img_path)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):

        img = np.load(self.img_path+self.fnames[idx])
        label = torch.tensor(int(self.fnames[idx].split('_')[-1].split('.')[0]))
        mask = np.load(self.label_path+self.fnames[idx])

        img1 = self.img_transform(self.preprocess_numpy(img, True))
        img2 = self.img_transform(self.preprocess_numpy(img, False))

        return img1, img2, label, mask

    def preprocess_numpy(self, x, wide):
        if wide:
            l_bound = -200
            u_bound = 300
        else:
            l_bound = 50
            u_bound = 150

        x = np.clip(x, l_bound, u_bound)
        x = x + 200
        x = x / 500
        x = x*255.0
        x = x.astype(np.uint8)
        x = np.stack((x,)*3, axis=-1)
        return x


class LiverUnn(Dataset):
    def __init__(self):

        self.root_path = 'set_path_to_data'

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        augmentation = [
            T.ToPILImage(),
            T.Resize(224),
            T.ToTensor(),
        ]

        self.img_path = ''.join([self.root_path, 'set_path'])
        self.label_path = ''.join([self.root_path, 'set_path'])

        self.img_transform = T.Compose(augmentation)
        self.fnames = listdir(self.img_path)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):

        img = np.load(self.img_path+self.fnames[idx])
        label = torch.tensor(int(self.fnames[idx].split('_')[-1].split('.')[0]))
        mask = np.load(self.label_path+self.fnames[idx])

        img1 = self.img_transform(self.preprocess_numpy(img, True))
        img2 = self.img_transform(self.preprocess_numpy(img, False))

        return img1, img2, label, mask

    def preprocess_numpy(self, x, wide):
        if wide:
            l_bound = -200
            u_bound = 300
        else:
            l_bound = 50
            u_bound = 150

        x = np.clip(x, l_bound, u_bound)
        x = x + 200
        x = x / 500
        x = x*255.0
        x = x.astype(np.uint8)
        x = np.stack((x,)*3, axis=-1)
        return x
