import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Union

from peak_detector.sslm.model import SSLModel


class SSLEvaluator:
    def __init__(
        self,
        valid_dl: DataLoader,
        state_dict_path: str,
        channels: int,
        device: torch.device,
        threshold=0.5,
    ) -> None:
        self.dl = valid_dl
        self.mask_height = 195
        self.mask_width = 487
        self.C = channels
        self.mAP = 0
        self.device = device

        self.threshold = threshold

        # initialise model in given state, freeze weights
        self.model = SSLModel(2)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(state_dict_path, map_location=device))
        self.model.eval()

    @torch.no_grad()
    def compute_mean_iou(self, return_average=True) -> Union[np.float64, list]:
        """Find the average IoU score over an entire dataset

        Args:
            return_average (bool, optional): whether to return the average value or list of all values. Defaults to True.

        Returns:
            Union[np.float64, list]: either the single average value (default) or the list of all calculated IoUs
        """
        avg_iou = []
        for batch_imgs, batch_targets in self.dl:
            valid_imgs = [valid_img.to(self.device) for valid_img in batch_imgs]
            valid_targets = [
                {key: val.to("cpu") for key, val in t.items()} for t in batch_targets
            ]
            pred_targets = self.model(valid_imgs)

            # loaded back into cpu for evaluation
            for i in range(len(valid_imgs)):
                pred = pred_targets[i].permute(1, 2, 0).contiguous().view(-1, self.C)
                ignore, pred_slice = torch.max(pred, 1)
                pred_mask, peak_predicted = self.slice_to_masks(pred_slice)
                pred_mask = torch.from_numpy(pred_mask)
                if peak_predicted:
                    iou = self.compute_IoU(valid_targets[i]["masks"], pred_mask)
                    avg_iou.append(iou.cpu().item())
                elif not peak_predicted and len(valid_targets[i]["masks"]) == 0:
                    # perfect iou if correctly predicts nothing
                    avg_iou.append(1)
                else:
                    # 0 iou if peaks are present which are missed
                    avg_iou.append(0)

        if return_average:
            return np.average(avg_iou)
        else:
            return avg_iou

    def make_histogram(self, save_file: str):
        """Generate a histogram of IoU scores for all images in the dataset

        Args:
            save_file (str): file to save the generated histogram to (defaults as png)
        """
        ious = self.compute_mean_iou(return_average=False)
        plt.hist(ious, 20, range=(0, 1))
        plt.xlabel("IoU")
        plt.ylabel("Counts")
        plt.savefig(save_file)
        plt.show()
        print(f"Saving histogram to {save_file}")

        return np.average(ious)

    def compute_IoU(self, valid_masks: list[Tensor], pred_masks: Tensor) -> Tensor:
        """calculate the intersection over union score for the ground truth masks and the prediction

        Args:
            valid_masks (list[Tensor]): list of masks given as ground truth label
            pred_masks (Tensor): mask generated by the model

        Returns:
            Tensor: IoU score - float value between 0 and 1
        """
        ## combine all the masks so all areas of peak are included
        all_v_masks = torch.zeros(
            (self.mask_height, self.mask_width), dtype=torch.float
        ).to("cpu")
        for mask in valid_masks:
            all_v_masks = all_v_masks | mask

        ## compute IoU of the total masks
        # flatten masks to 1D
        v_mask = all_v_masks.flatten()
        p_mask = pred_masks.flatten()

        v_area = torch.sum(v_mask)
        p_area = torch.sum(p_mask)

        intersection = torch.dot(v_mask, p_mask)
        union = v_area + p_area - intersection

        return (intersection / union).cpu()

    def precision_recall(self) -> tuple[float, float]:
        """Calculate precision and recall scores for the dataset (at the specified threshold)
        TO DO:
        * test precision recall calculations and generate curve at different thresholds

        Returns:
            tuple[float, float]: precision and recall scores
        """
        true_pos = 0
        false_pos = 0
        false_neg = 0

        for batch_imgs, batch_targets in self.dl:
            for i in range(len(batch_imgs)):
                image = batch_imgs[i].to(self.device)
                pred_target = self.model(image.unsqueeze(dim=0))[0]

                pred = pred_target.permute(1, 2, 0).contiguous().view(-1, self.C)
                ignore, pred_slice = torch.max(pred, 1)
                pred_slice = pred_slice.reshape((self.model.width, self.model.height))
                pred_mask, peak_predicted = self.slice_to_masks(
                    pred_slice.to(torch.device("cpu"))
                )
                pred_mask = torch.from_numpy(pred_mask)

                if peak_predicted:
                    iou = self.compute_IoU(batch_targets[i]["masks"], pred_mask)
                    if iou < 0.5:
                        false_pos += 1
                    elif iou >= 0.5:
                        true_pos += 1
                else:
                    if len(batch_targets[i]["masks"]) == 1:
                        true_pos += 1  # does this need to be here or should this be true negative??
                    else:
                        false_neg += 1

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        print(f"Precision: {precision} | Recall: {recall}")
        return precision, recall

    def slice_to_masks(self, slice: np.ndarray) -> tuple[np.ndarray, bool]:
        """Convert the channel outputs into a binary mask
        TO DO:
        * change to take peak channels and background channels instead

        Args:
            slice (np.ndarray): input array where each pixel is labelled by its most likely channel

        Returns:
            tuple[np.ndarray, bool]: binary mask and boolean determining if any peak is present
        """
        flat = slice.flatten()
        counts = np.bincount(flat)
        most_common_label = counts.argmax()
        mask = np.ones((self.mask_height, self.mask_width))
        peak_predicted = False

        for i in range(self.mask_height):
            for j in range(self.mask_width):
                if slice[i, j] == most_common_label:
                    mask[i, j] = 0
                else:
                    peak_predicted = True

        return mask, peak_predicted
