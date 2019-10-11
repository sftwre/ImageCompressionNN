import os
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np


TRAIN_TRANSFORMS_256 = transforms.Compose([
    transforms.RandomResizedCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # from [0, 1] to [-1, 1]
])

TEST_TRANSFORMS_256 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def is_image_file(img):
    return any(img.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".JPEG"])

def load_image(path):
    return Image.open(path).convert('RGB')

def get_training_set(train_path):
    return DatasetFromFolder(train_path)

    # path = Path(train_path)
    #
    # def _loader():
    #     count = 0
    #
    #     to_yield = np.zeros((batch_size, 256, 256, 3))
    #
    #     for img in path.glob("**/*.jpg"):
    #
    #         img = np.asarray(Image.open(img).resize((256, 256)))
    #
    #         count += 1
    #
    #         if count == batch_size and batch_size > 1:
    #             np_imgs = np.transpose(to_yield, (0, 3, 1, 2)) / 256.
    #             yield torch.Tensor(np_imgs).cuda()
    #             count = 1
    #             to_yield = np.zeros((batch_size, 256, 256, 3))
    #             to_yield[0, :, :, :] = img
    #         else:
    #             to_yield[0, :, :, :] = img
    #
    #     np_imgs = np.transpose(to_yield, (0, 3, 1, 2)) / 256.
    #     yield torch.Tensor(np_imgs).cuda()
    #
    # return _loader



class DatasetFromFolder(torch.utils.data.Dataset):
    """
        Loads dataset from a given directory
    """

    def __init__(self, train_path):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [os.path.join(train_path, img) for img in os.listdir(train_path) if is_image_file(img)]

    def __getitem__(self, item):

        img = load_image(self.image_filenames[item])
        img = TRAIN_TRANSFORMS_256(img)

        return img

    def __len__(self):
        return len(self.image_filenames)
        

