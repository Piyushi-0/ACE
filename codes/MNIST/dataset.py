"""dataset.py"""

import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def return_data(args):
	name = args.dataset
	dset_dir = args.dset_dir
	batch_size = args.batch_size
	num_workers = args.num_workers
	image_size = args.image_size
    # assert image_size == 64, 'currently only image size of 64 is supported'
	root = os.path.expanduser('~/data/mnist')
	transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),])
	train_kwargs = {'root':root, 'transform':transform}
	dset = CustomImageFolder

	data_path = os.path.expanduser('~/data/mnist')
        
        kwargs = {}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_path, train = True, download = True, transform = transforms.ToTensor()),
            batch_size = args.batch_size, shuffle = True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_path, train = False, transform = transforms.ToTensor()),
            batch_size = 1, shuffle = True, **kwargs)

	data_loader = train_loader

	return data_loader

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),])

    dset = CustomImageFolder(root, transform)
    loader = DataLoader(dset,
                       batch_size=32,
                       shuffle=True,
                       num_workers=1,
                       pin_memory=False,
                       drop_last=True)

    images1 = iter(loader).next()
    import ipdb; ipdb.set_trace()
