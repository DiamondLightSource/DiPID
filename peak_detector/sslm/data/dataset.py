from typing import Any, cast
from torch import Tensor
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
import torchvision
from matplotlib import pyplot as plt
from matplotlib.path import Path
import numpy as np
import h5py
import pandas as pd
import json


class PeakDataset(Dataset):
    def __init__(self, folder: str, nxs_nos: list, train: bool):
        """Set up the dataset - this is for unsupervised training so no labels are needed

        Args:
            nxs_folder (str): folder containing the nexus file
            nxs_nos (list): list of all nexus files used in the dataset
            train (bool): whether this dataset is used for training or not
        """
        self.folder = folder
        self.root_folder = "peak_detector/nexus"
        self.train = train

        self.nexus_numbers: dict[str, int] = {k: 0 for k in nxs_nos}
        # work out how many images are in each nexus file
        for number in nxs_nos:
            filename = f"{self.root_folder}/{self.folder}/{number}.nxs"
            with h5py.File(filename) as file:
                volume = cast(h5py.Dataset, file["entry1/pil3_100k/data"])
                self.nexus_numbers[number] = volume.shape[0]
        self.height = 487
        self.width = 195
        self.transforms = self.get_transforms()

    def __len__(self) -> int:
        count = 0
        for amount in self.nexus_numbers.values():
            count += amount
        return count

    def __getitem__(self, idx: int) -> tuple[Tensor, dict]:
        """Returns a single image from the dataset

        Args:
            idx (int): Index of the image to be returned - unique to each

        Returns:
            tuple[Tensor, dict]: the tensor of the image data as well as the identifying nexus file and image index
        """

        # convert index into nexus number and image number
        nxs_no, img_no = "0", 0
        total_imgs = 0
        for item in self.nexus_numbers.items():
            total_imgs += item[1]
            if idx < total_imgs:
                nxs_no = item[0]
                img_no = idx + item[1] - total_imgs
                break
        # print(f"Index: {idx}; nxs: {nxs_no}; img: {image_no}")

        # load in and normalise the image
        image = self.read_image(nxs_no, img_no)
        image = (image - image.min()) / (image.max() - image.min())

        if self.transforms is not None:
            image = self.transforms(image)

        # expected format has channels as the first column so manually add extra dimension
        image = image.unsqueeze(dim=0)

        label = {"nxs_no": nxs_no, "img_no": img_no}
        return image, label

    def read_image(self, nxs_no: str, image_no: int) -> Tensor:
        """Load image data in from the nexus file and return it as a tensor
        TO DO:
        * generalise image reading so merlin and fast shutter data can also be read?


        Args:
            nxs_no (str): scan number of the nexus file
            image_no (int): index of the image in the scan

        Returns:
            Tensor: image data from the nexus file
        """
        filename = f"{self.root_folder}/{self.folder}/{nxs_no}.nxs"
        with h5py.File(filename) as file:
            volume = cast(h5py.Dataset, file["entry1/pil3_100k/data"])
            image = self.remove_dead_pixels(volume[image_no, :, :])
            return torch.from_numpy(image)

    def get_transforms(self) -> Any:
        """Compiles a list of transformations for the train dataset or does nothing for the test set

        Returns:
            Any: either a list of transforms to apply or empty return
        """
        torchvision.disable_beta_transforms_warning()
        if self.train:
            return transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                ]
            )
        else:
            return None

    def remove_dead_pixels(self, image: np.ndarray) -> np.ndarray:
        """The detector has some broken pixels, so we replace these with an average of their neighbours

        Args:
            image (np.ndarray): image array with broken pixels

        Returns:
            np.ndarray: edited array with averages
        """
        pixels: list[tuple[int, int]] = [
            (40, 111),
            (46, 301),
            (46, 302),
            (75, 184),
            (75, 185),
            (75, 187),
            (97, 153),
            (98, 153),
            (157, 89),
            (179, 450),
            (190, 473),
        ]
        for pixel in pixels:
            x = pixel[0]
            y = pixel[1]
            count: int = 0
            temp: int = 0
            neighbours = [
                image[x + 1, y],
                image[x - 1, y],
                image[x, y + 1],
                image[x, y - 1],
            ]
            for neighbour in neighbours:
                # dead pixels have the value -2
                if neighbour > 0:
                    temp += neighbour
                    count += 1
            if count > 1:
                image[x, y] = np.divide(temp, count)
            else:
                image[x, y] = temp
        return image


class ValidationDataset(PeakDataset):
    def __init__(self, csv_file: str, nxs_folder: str, nxs_nos: list, train=False):
        """This dataset also returns labels for evaluation purposes

        Args:
            csv_file (str): file containing the ground truth labels
            nxs_folder (str): folder containing the nexus files
            nxs_nos (list): list of nexus scans included in the dataset
            train (bool, optional): this dataset should only be used for test/validation sets. Defaults to False
        """
        super().__init__(nxs_folder, nxs_nos, train)
        self.csv_file = csv_file
        self.train = False

    def __getitem__(self, idx: int) -> tuple[Tensor, dict]:
        """Returns a single image from the dataset and its ground truth labels

        Args:
            idx (int): Index of the image to be returned - unique to each

        Returns:
            tuple[Tensor, dict]: the tensor of the image data and a dictionary of its ground truth labels
        """
        image, label = super().__getitem__(idx)
        nxs = str(label["nxs_no"])
        img = str(label["img_no"])
        full_label = nxs + "-" + img

        # find masks for the given nexus file and image number
        df = pd.read_csv(self.csv_file)
        # find all instances in the csv (images may have multiple rows)
        indices: list[int] = df.index[
            df["subject_data"].str.contains(full_label)
        ].tolist()

        masks = []
        for index in indices:
            annotations = json.loads(df["annotations"][index])
            if len(annotations) > 1:
                for polygon in annotations[1]["value"]:
                    vertices = self.get_polygon_vertices(polygon)
                    masks.append(self.make_mask(vertices))
        label["masks"] = masks

        return image, label

    def make_mask(self, point_tuples: list[tuple[float, float]]) -> Tensor:
        """Converts a list of co-ordinates into a binary mask of regions inside and outside the polygon they enclose

        Args:
            point_tuples (list[tuple[float, float]]): list of co-ordinates describing a closed polygon

        Returns:
            Tensor: binary mask in the same dimensions as the image
        """
        mask = torch.zeros(size=(self.width, self.height), dtype=torch.bool)
        path = Path(point_tuples)
        for x in range(self.width):
            for y in range(self.height):
                if path.contains_point((y, x), radius=0.5):
                    mask[x, y] = True
        return mask

    def get_polygon_vertices(
        self, polygon: dict[str, Any]
    ) -> list[tuple[float, float]]:
        points = polygon["points"]
        vertices = [(point["x"], point["y"]) for point in points]
        return vertices


if __name__ == "__main__":
    # write script to automatically collect all nexus files?
    # could iterate over a folder
    # would need more validation for loading data in as some might be merlin data etc
    nxs_nos = [
        "794625",
        "794661",
        "794663",
        "794689",
        "794673",
        "794710",
        "794910",
        "794916",
        "794927",
    ]
    dataset = PeakDataset("nexus/mm24570-1", nxs_nos, False)
    print(f"Length: {dataset.__len__()}")
    image = dataset.__getitem__(61)[0]
    image = image.permute(1, 2, 0)
    plt.imshow(image, alpha=0.9)
    plt.show()
