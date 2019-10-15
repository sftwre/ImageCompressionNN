import os
import torch
from torchvision import transforms
from PIL import Image


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
        

