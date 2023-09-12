import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from peak_detector.tlm.data.dataset import PeakDataset


def get_dataloaders(
    batch_size: int, train_ratio: float, csv_file: str, nxs_folder: str
) -> tuple[DataLoader, DataLoader]:
    """Split the data into test and train sets and put these into dataloaders for easy batching etc.

    Args:
        batch_size (int): Number of images per batch in the train set
        train_ratio (float): Fraction of images to assign to the train set (in the range 0 -1)
        csv_file (str): File containing all datapoints and label information
        nxs_folder (str): Folder containing detector images

    Returns:
        tuple[DataLoader, DataLoader]: train and test dataloaders to be used directly in a training loop
    """
    full_training_data = PeakDataset(csv_file, nxs_folder, True)
    full_test_data = PeakDataset(csv_file, nxs_folder, False)

    # randomise list of indices and split into test and train:
    total_length: int = full_training_data.__len__()
    indices: list[int] = torch.randperm(total_length).tolist()
    split_idx = int(total_length * train_ratio)
    train_idx, test_idx = np.split(indices, [split_idx])

    training_data = Subset(full_training_data, train_idx.tolist())
    test_data = Subset(full_test_data, test_idx.tolist())

    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=maskrcnn_collate_function,
    )
    test_dataloader = DataLoader(
        test_data, batch_size=5, shuffle=False, collate_fn=maskrcnn_collate_function
    )

    return train_dataloader, test_dataloader


def maskrcnn_collate_function(batch):
    # need to define our own collate function to get around having different amounts of peaks in each scan
    return tuple(zip(*batch))


if __name__ == "__main__":
    get_dataloaders(32, 0.8, "sample.csv", "nexus/mm24570-1")
