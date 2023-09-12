import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from typing import cast
from matplotlib.colors import Normalize
import h5py

from peak_detector.sslm.model import SSLModel
from peak_detector.sslm.utils.loss_utils.centres import get_centres


class SSLPredictor:
    def __init__(self, state_dict_path: str) -> None:
        self.state_dict_path = state_dict_path

        self.model = SSLModel(2)
        self.model.load_state_dict(
            torch.load(state_dict_path, map_location=torch.device("cpu"))
        )
        self.model.eval()
        self.height = self.model.height
        self.width = self.model.width

    def get_image(self, folder: str, nxs_no: str, img_no: int) -> Tensor:
        """_summary_

        Args:
            folder (str): folder containing the nexus file
            nxs_no (str): scan number of the specified nexus file
            img_no (int): image index within the scan

        Returns:
            Tensor: specified image from the nexus file
        """
        filename = f"peak_detector/nexus/{folder}/{nxs_no}.nxs"
        with h5py.File(filename) as file:
            volume = cast(h5py.Dataset, file["entry1/pil3_100k/data"])
            image = self.remove_dead_pixels(volume[img_no, :, :])
            image = torch.from_numpy(image)
            # normalise image
            image = (image - image.min()) / (image.max() - image.min())
            return image.unsqueeze(dim=0)

    def remove_dead_pixels(self, image: np.ndarray) -> np.ndarray:
        """The detector has dead pixels so we replace these values with the average value of all neighbouring pixels

        Args:
            image (np.ndarray): original image from the detector

        Returns:
            np.ndarray: edited image with averages instead of broken pixels
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
            if count > 1:
                image[x, y] = np.divide(temp, count)
            else:
                image[x, y] = temp
        return image

    @torch.no_grad()
    def predict(self, folder: str, nxs_no: str, img_no: int, show_img=False) -> Tensor:
        """Generates the model predicted masks for the given scan from its nexus file

        Args:
            folder (str): folder containing the nexus file
            nxs_no (str): scan number of the specified nexus file
            img_no (int): image index within the scan
            show_img (bool, optional): whether to display the output after generation. Defaults to False.

        Returns:
            Tensor: _description_
        """
        image = self.get_image(folder, nxs_no, img_no)

        outs = self.model(image.unsqueeze(dim=0))[0]
        output = outs.permute(1, 2, 0).contiguous().view(-1, self.model.out_channels)
        ignore, pred = torch.max(output, 1)
        pred = pred.reshape((self.width, self.height, 1))
        labels = np.unique(pred)
        # print(labels)
        # try to combine background pixels
        pred = self.merge_background(pred)

        image = np.asarray(image.permute(1, 2, 0))
        log_image = np.log(image + 0.00001)

        norm = Normalize(vmin=pred.min(), vmax=pred.max())
        norm_pred = norm(pred)

        if show_img:
            self.show_output(image, log_image, norm_pred)

        return norm_pred

    def merge_background(self, pred: Tensor) -> Tensor:
        """Relabel background channels as 0 and peak channels as 1

        Args:
            pred (Tensor): prediction with multiple channels

        Returns:
            Tensor: binary mask of peaks
        """
        background = [20, 25, 32, 38, 57, 75, 80, 91]
        peaks = [13, 66]
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                if pred[i, j] in peaks:
                    pred[i, j] = 1
                else:
                    pred[i, j] = 0
        labels = np.unique(pred)
        # print(labels)
        return pred

    @torch.no_grad()
    def identify_background(self, folder: str, nxs_no: str, img_no: int) -> None:
        """Loop over each predicted channel to manually identify if it is background or peak

        Args:
            folder (str): folder containing the nexus file
            nxs_no (str): scan number of the specified nexus file
            img_no (int): image index within the scan
        """
        model = SSLModel(2)
        model.load_state_dict(
            torch.load(self.state_dict_path, map_location=torch.device("cpu"))
        )
        model.eval()

        image = self.get_image(folder, nxs_no, img_no)

        outs = model(image.unsqueeze(dim=0))[0]
        output = outs.permute(1, 2, 0).contiguous().view(-1, model.out_channels)
        ignore, pred = torch.max(output, 1)
        pred = pred.reshape((model.width, model.height, 1))
        labels = np.unique(pred)

        image = image.permute(1, 2, 0)
        log_image = np.log(image + 0.0001)

        norm = Normalize(vmin=pred.min(), vmax=pred.max())
        for label in labels:
            print(label)
            cluster = np.where(pred == label, label, 0)
            fig = plt.figure()
            fig.add_subplot(2, 2, 1)
            plt.imshow(image)
            fig.add_subplot(2, 2, 2)
            norm_pred = norm(cluster)
            plt.imshow(norm_pred, cmap="viridis")
            #    plt.annotate(f"Number of clusters: {len(labels)}", (10, 30))
            fig.add_subplot(2, 2, 3)
            plt.imshow(log_image)
            plt.show()

    def show_centres(
        self, folder: str, nxs_no: str, img_no: int
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """Convert the calculated centres to pixel co-ordinates for the two peak channels

        Args:
            folder (str): folder containing the nexus file
            nxs_no (str): scan number of the specified nexus file
            img_no (int): image index within the scan

        Returns:
            tuple[tuple[float], tuple[float]]: co-ordinates of the centres of the two peak channels
        """
        image = self.get_image(folder, nxs_no, img_no)

        outs = self.model(image.unsqueeze(dim=0))[0]
        centres = get_centres(
            outs,
            torch.device("cpu"),
            100,
        )
        print(centres[13], centres[66])

        a = ((centres[13][0] + 1) / 2) * self.height
        b = ((centres[13][0] + 1) / 2) * self.width
        c = ((centres[66][0] + 1) / 2) * self.height
        d = ((centres[66][0] + 1) / 2) * self.width
        print(f"13: [{a}, {b}] | 66: [{c}, {d}]")
        return (a, b), (c, d)

    def show_output(
        self, image: np.ndarray, log_image: np.ndarray, prediction: torch.Tensor
    ) -> None:
        """Display the original image and the prediction

        Args:
            image (np.ndarray): initial input image
            log_image (np.ndarray): input image with a log scaling applied
            prediction (torch.Tensor): binary mask predicted by the model
        """
        fig = plt.figure()
        plt.suptitle("Self-Supervised Prediction")
        fig.add_subplot(2, 2, 1)
        plt.imshow(image, cmap="viridis")
        plt.imshow(prediction, cmap="Greys", alpha=0.25)
        fig.add_subplot(2, 2, 2)
        plt.imshow(log_image, cmap="viridis")
        plt.imshow(prediction, cmap="Greys", alpha=0.25)
        fig.add_subplot(2, 2, 3)
        plt.imshow(image, cmap="viridis")
        fig.add_subplot(2, 2, 4)
        plt.imshow(log_image, cmap="viridis")
        plt.show()
