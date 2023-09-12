from typing import Any, cast
from torch import Tensor, zeros
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms.v2 as transforms
import torchvision
from matplotlib.path import Path
from matplotlib import pyplot as plt
import numpy as np
import json
import h5py


class PeakDataset(Dataset):
    def __init__(
        self, csv_file: str, nxs_folder: str, train: bool
    ):
        """Set up the file for the data and other necessary values
        TO DO:
        * generalise image dimensions

        Args:
            csv_file (str): file with labelled data
            nxs_folder (str): folder containing the nexus files for this data
            train (bool): whether this dataset will be used for training or not
        """
        self.peak_df = pd.read_csv(csv_file, header=0)
        self.train = train
        self.nxs_folder = nxs_folder
        self.height = 487
        self.width = 195
        self.transforms = self.get_transforms()

    def __len__(self) -> int:
        return len(self.peak_df)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict[str, Any]]:
        """Return a single item from the dataset with the associated labels

        Args:
            idx (int): Index of the requested data point - unique to each row in the csv file

        Returns:
            tuple[Tensor, dict[str, Any]]: the requested image tensor as well as a dictionary of labels including ground truth masks and boxes
        """
        subject_data = json.loads(self.peak_df["subject_data"][idx])
        subject_id = self.peak_df["subject_ids"][idx]
        image_no = subject_data[str(subject_id)]["Image Number"]
        image = self.read_image(image_no)
        
        #normalise and add extra dimension so input is compatible with the ResNet backbone
        image = torch.unsqueeze(image, dim=0)
        image = (image - image.min()) / (image.max() - image.min())

        annotations = json.loads(self.peak_df["annotations"][idx])
        all_masks = []
        labels = []
        bboxes = []
        if len(annotations) > 1:
            for polygon in annotations[1]["value"]:
                vertices = self.get_polygon_vertices(polygon)
                polygon_mask = self.make_mask(vertices)
                bbox = self.make_bounding_box(vertices)
                all_masks.append(polygon_mask)
                # 0 is reserved for background, 1 represents a peak => model must be initialised with two classes
                labels.append(1)
                bboxes.append(bbox)
        else:
            # dataset may also contain images with no peaks, so we return empty labels and a background tag
            labels = [0]
            bboxes = np.zeros((0, 4), dtype=np.float32)

        if self.transforms is not None:
            # transforms v2 automatically applies the same transformations to the labels as the image
            transformed = self.transforms(
                image, {"masks": all_masks, "boxes": bboxes, "labels": labels}
            )
            image = transformed[0]
            all_masks = transformed[1]["masks"]
            bboxes = transformed[1]["boxes"]
            labels = transformed[1]["labels"]

        # convert from a list of masks to a tensor
        mask_tensor = torch.zeros((len(all_masks), self.width, self.height))
        for i in range(len(all_masks)):
            mask_tensor[i, :, :] = all_masks[i]

        # expected return format is img, dict{labels}
        target = {}
        target["masks"] = mask_tensor
        # expected format for bboxes is a [n, 4] tensor
        target["boxes"] = torch.as_tensor(bboxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        nxs_no, img_idx = image_no.split("-")
        target["image_no"] = torch.as_tensor(int(nxs_no), dtype=torch.int)
        target["image_idx"] = torch.as_tensor(int(img_idx), dtype=torch.int)

        return image, target

    def make_mask(self, point_tuples: list[tuple[float, float]]) -> Tensor:
        """Converts a list of co-ordinates into a binary mask of the regions inside and outside

        Args:
            point_tuples (list[tuple[float, float]]): list of co-ordinates describing a closed polygon 

        Returns:
            Tensor: _description_
        """
        mask = zeros(size=(self.width, self.height), dtype=torch.bool)
        path = Path(point_tuples)
        for x in range(self.width):
            for y in range(self.height):
                if path.contains_point(
                    (y, x), radius=0.5
                ):
                    mask[x, y] = True
        return mask

    def make_bounding_box(self, point_tuple: list[tuple[float, float]]) -> list[float]:
        """Converts a list of co-ordinates into a box which encloses all the points

        Args:
            point_tuple (list[tuple[float, float]]): list of co-ordinates describing a closed polygon 

        Returns:
            list[float]: maximum and minimum x and y values of the box around the region
        """
        xmin, ymin = float("inf"), float("inf")
        xmax, ymax = 0, 0
        for point in point_tuple:
            xmax = max(xmax, point[0])
            ymax = max(ymax, point[1])
            xmin = min(xmin, point[0])
            ymin = min(ymin, point[1])
        return [xmin, ymin, xmax, ymax]

    def get_polygon_vertices(
        self, polygon: dict[str, Any]
    ) -> list[tuple[float, float]]:
        points = polygon["points"]
        vertices = [(point["x"], point["y"]) for point in points]
        return vertices

    def read_image(self, image_no: str) -> Tensor:
        """Read the scan image data from the nexus file and load into a tensor

        Args:
            image_no (str): index in the scan of the desired image

        Returns:
            Tensor: single image from the detector
        """
        nxs_no, idx = image_no.split("-")
        filename = f"peak_detector/nexus/{self.nxs_folder}/{nxs_no}.nxs"
        with h5py.File(filename) as file:
            volume = cast(h5py.Dataset, file["entry1/pil3_100k/data"])
            image = self.remove_dead_pixels(volume[int(idx), :, :])
            return torch.from_numpy(image)

    
    def get_transforms(self) -> Any:
        """Compile the list of transformations to apply to image data before training
        No transformations are applied to test data

        Returns:
            Any: Either a list of transformations or an empty return
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
        """Some pixels in the detector are broken, so replace these with an average of their neighbours

        Args:
            image (np.ndarray): array containing detector image

        Returns:
            np.ndarray: edited array with broken pixels averaged out
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
                if neighbour > 0:  # dead pixels have the value -2
                    temp += neighbour
                    count += 1
            # need to account for pixels with 0 as a neighbor
            if count > 1:
                image[x, y] = np.divide(temp, count)
            else:
                image[x, y] = temp
        return image


if __name__ == "__main__":
    dataset = PeakDataset("sample.csv", "nexus/mm24570-1", True)

    item_list = dataset.__getitem__(3)
    image, masks = item_list[0], item_list[1]["masks"]

    for mask in masks:
        plt.imshow(image, alpha=0.9)
        plt.imshow(mask, alpha=0.5, cmap="Greys")
        plt.show()
