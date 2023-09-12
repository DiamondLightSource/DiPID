import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from peak_detector.sslm.data.dataset import PeakDataset, ValidationDataset


def get_dataloaders(
    batch_size: int, train_ratio: float, nxs_folder: str, nxs_nos: list, csv_file: str
) -> tuple[DataLoader, DataLoader]:
    """Split the data into test and train sets and put these into dataloaders for easy batching etc

    Args:
        batch_size (int): Number of images per batch in the train set
        train_ratio (float): Fraction of images to assign to the train set (in the range 0 -1)
        nxs_folder (str): Folder containing detector images
        nxs_nos (list): list of nexus scans in the dataset
        csv_file (str): File containing all datapoints and label information

    Returns:
        tuple[DataLoader, DataLoader]: train and test dataloaders to be used directly in a training loop
    """

    full_training_data = PeakDataset(nxs_folder, nxs_nos, True)
    full_test_data = ValidationDataset(csv_file, nxs_folder, nxs_nos, False)

    # need to split dataset into test and train values
    total_length: int = full_training_data.__len__()
    # possiblity to fix random seed
    # randomise list of indices and split into test and train:
    indices: list[int] = torch.randperm(total_length).tolist()
    split_idx = int(total_length * train_ratio)
    train_idx, test_idx = np.split(indices, [split_idx])
    training_data = Subset(full_training_data, train_idx.tolist())
    test_data = Subset(full_test_data, test_idx.tolist())

    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_data, batch_size=5, shuffle=False, collate_fn=mask_collate_function
    )

    return train_dataloader, test_dataloader


def mask_collate_function(batch):
    # need to define our own collate function to get around having different amounts of peaks in each scan
    return tuple(zip(*batch))
