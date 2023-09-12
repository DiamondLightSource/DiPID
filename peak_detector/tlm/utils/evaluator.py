import torch
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Union

from peak_detector.tlm.model import TLModel


## inspired by: https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/utils.py
class TLMEvaluator:
    def __init__(
        self,
        valid_dl: DataLoader,
        state_dict_path: str,
        device: torch.device,
        threshold=0.4,
    ) -> None:
        """Load model from given weights
        TO DO:
        * generalise image dimensions

        Args:
            valid_dl (DataLoader): dataset to evaluate the model on
            state_dict_path (str): file containing model weights
            device (torch.device): set to GPU if GPU processing is available
            threshold (float, optional): confidence threshold for accepting a prediction. Defaults to 0.4.
        """
        self.dl = valid_dl
        self.mask_height = 195
        self.mask_width = 487
        self.mAP = 0
        self.device = device
        self.threshold = threshold

        # initialise model in given state, freeze weights
        self.model = TLModel(2).model
        self.model.load_state_dict(
            torch.load(state_dict_path, map_location=self.device)
        )
        self.model.eval()

    def compute_IoU(
        self, valid_masks: list[Tensor], pred_masks: list[Tensor]
    ) -> Tensor:
        """calculate the intersection over union score for the ground truth masks and the prediction

        Args:
            valid_masks (list[Tensor]): list of masks given as ground truth label
            pred_masks (list[Tensor]): list of masks generated by the model

        Returns:
            Tensor: IoU score - float value between 0 and 1
        """
        ## combine all the masks so all areas of peak are included
        all_v_masks = torch.zeros(
            (self.mask_height, self.mask_width), dtype=torch.float
        ).to("cpu")
        # with 0.5 threshold for object detection in a pixel (could set as variable but this is standard)
        for mask in valid_masks:
            for i in range(self.mask_height):
                for j in range(self.mask_width):
                    if mask[i, j] > 0.5:
                        mask[i, j] = 1
                    else:
                        mask[i, j] = 0
            all_v_masks = all_v_masks + mask
        all_p_masks = torch.zeros(
            (self.mask_height, self.mask_width), dtype=torch.float
        ).to("cpu")
        for mask in pred_masks:
            for i in range(self.mask_height):
                for j in range(self.mask_width):
                    if mask[i, j] > 0.5:
                        mask[i, j] = 1
                    else:
                        mask[i, j] = 0
            all_p_masks = all_p_masks + mask

        ## compute IoU of the total masks
        # flatten masks to 1D so we can take a dot product
        v_mask = all_v_masks.flatten()
        p_mask = all_p_masks.flatten()

        v_area = torch.sum(v_mask)
        p_area = torch.sum(p_mask)

        intersection = torch.dot(v_mask, p_mask)
        union = v_area + p_area - intersection

        return (intersection / union).cpu()

    def compute_mean_iou(self, return_average=True) -> Union[np.float64, list]:
        """Find the average IoU score over an entire dataset

        Args:
            return_average (bool, optional): whether to return the average value or list of all values. Defaults to True.

        Returns:
            Union[np.float64, list]: either the single average value (default) or the list of all calculated IoUs
        """
        avg_iou = []
        for valid_imgs, valid_targets in self.dl:
            valid_imgs = [valid_img.to(self.device) for valid_img in valid_imgs]
            valid_targets = [
                {key: val.to("cpu") for key, val in t.items()} for t in valid_targets
            ]

            pred_targets = self.model(valid_imgs)
            pred_targets = [
                {key: val.to("cpu") for key, val in t.items()} for t in pred_targets
            ]

            for i in range(len(valid_imgs)):
                all_pred_masks = pred_targets[i]["masks"]
                scores = pred_targets[i]["scores"]
                pred_masks = []
                # remove all masks with confidence scores below the threshold
                for j, score in enumerate(scores):
                    if score > self.threshold:
                        pred_masks.append(torch.squeeze(all_pred_masks[j], dim=0))

                iou = self.compute_IoU(valid_targets[i]["masks"], pred_masks)
                avg_iou.append(iou)

        if return_average:
            return np.average(avg_iou)
        else:
            return avg_iou

    def make_histogram(self, save_file: str):
        """Generate a histogram of IoU scores for all images in the dataset

        Args:
            save_file (str): file to save the generated histogram to (defaults as png)
        """
        ious = self.compute_mean_iou(False)
        plt.hist(ious, 10, range=(0, 1))
        plt.savefig(save_file)
        plt.show()

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
        for valid_imgs, valid_targets in self.dl:
            valid_imgs = [valid_img.to(self.device) for valid_img in valid_imgs]
            valid_targets = [
                {key: val.to("cpu") for key, val in t.items()} for t in valid_targets
            ]

            pred_targets = self.model(valid_imgs)
            pred_targets = [
                {key: val.to("cpu") for key, val in t.items()} for t in pred_targets
            ]

            for i in range(len(valid_imgs)):
                all_pred_masks = pred_targets[i]["masks"]
                scores = pred_targets[i]["scores"]
                pred_masks = []
                # remove all masks below confidence threshold
                for j, score in enumerate(scores):
                    if score > self.threshold:
                        pred_masks.append(torch.squeeze(all_pred_masks[j], dim=0))

                if len(pred_masks) == 0:
                    if len(valid_targets[i]["masks"]) == 0:
                        true_pos += 1  # does this need to be here or should this be true negative??
                    else:
                        false_neg += 1
                else:
                    iou = self.compute_IoU(valid_targets[i]["masks"], pred_masks)
                    if iou > 0.5:
                        true_pos += 1
                    else:
                        false_pos += 1

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        print(f"Precision: {precision} | Recall: {recall}")
        return precision, recall

    def non_max_supression(self, targets: dict[str, Any], nms_threshold: float):
        """Removes predictions which have too much overlap by prioritising those with the highest confidence
        https://paperswithcode.com/method/non-maximum-suppression

        Args:
            targets (dict[str, Any]): dictionary of generated predicitions including masks and bounding boxes
            nms_threshold (float): level of overlap required to dispose of a prediction (between 0 and 1)

        Returns:
            _type_: _description_
        """
        removed = 0
        boxes = targets["boxes"]
        masks = targets["masks"]
        scores = np.asarray(targets["scores"])
        nms_targets = {
            "boxes": [],
            "masks": [],
            "scores": [],
        }
        while len(scores) > removed:
            max_idx = scores.argmax()
            nms_targets["boxes"].append(boxes[max_idx])
            nms_targets["masks"].append(masks[max_idx])
            nms_targets["scores"].append(scores[max_idx])

            removed += 1
            scores[max_idx] = 0

            for i in range(len(scores)):
                if scores[i] > 0:
                    iou = self.compute_IoU([masks[max_idx]], [masks[i]])
                    if iou > nms_threshold:
                        scores[i] = 0
                        removed += 1

        return nms_targets
