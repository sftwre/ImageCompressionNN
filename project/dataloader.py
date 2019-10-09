import os
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms, datasets
import numpy as np


# TODO get clarification on this
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

"""
    Dataset used for training and testing
"""
class Raise1K:

    def __init__(self, train_path, test_path):

        self.train_path = train_path
        self.test_path = test_path
        self.transform_train = TRAIN_TRANSFORMS_256
        self.transform_test = TEST_TRANSFORMS_256\


    def data_loader(self, workers, batch_size, num_train, num_test, train_indices_path, test_indices_path):


        print("===> Preparing Raise1K")
        transform_train, transform_test = self.transform_train, self.transform_test

        trainset = datasets.ImageFolder(self.train_path, transform_train)
        testset = datasets.ImageFolder(self.test_path, transform_test)


        if os.path.exists(test_indices_path):
            print("Using stored test indices")
            test_indices = np.load(test_indices_path)
        else:
            test_indices = list(range(250)) # size of test set
            np.random.shuffle(test_indices)
            np.save(test_indices_path, test_indices)

        if os.path.exists(train_indices_path):
            print("Using stored train indices")
            train_indices = np.load(train_indices_path)
        else:
            print(len(trainset))

            train_indices = list(range(len(trainset)))
            np.random.shuffle(train_indices)
            np.save(train_indices_path, train_indices)

        train_indices, valid_indices = train_indices[:num_train], test_indices[200:(200 + num_test)]
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(valid_indices)

        train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=workers, sampler=train_sampler)
        test_loader = DataLoader(testset, batch_size=3 * batch_size, sampler=val_sampler,
                                 num_workers=workers)

        return train_loader, test_loader





