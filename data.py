import os
import torch
from sklearn.model_selection import KFold
from torchvision import datasets, transforms


def load_datasets(dataset_path):
    image_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    train_transforms = transforms.Compose(image_transforms)
    test_transforms = transforms.Compose(image_transforms[1:])  # skip augmentations

    train_dataset = datasets.ImageFolder(
        os.path.join(dataset_path, "train"), transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(dataset_path, "val"), transform=train_transforms
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(dataset_path, "test"), transform=test_transforms
    )

    return train_dataset, val_dataset, test_dataset


def load_datasets_v2(dataset_path, n_folds=10, seed=0):
    dataset = datasets.ImageFolder(dataset_path)

    # transforms
    image_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.531, 0.478, 0.430], std=[0.251, 0.243, 0.241]),
    ]
    train_transforms = transforms.Compose(image_transforms)
    test_transforms = transforms.Compose(image_transforms[1:])

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for idx, (train_index, test_index) in enumerate(kf.split(dataset)):
        train = torch.utils.data.Subset(dataset, train_index)
        test = torch.utils.data.Subset(dataset, test_index)

        train.dataset.transform = train_transforms
        test.dataset.transform = test_transforms
        return train, test
