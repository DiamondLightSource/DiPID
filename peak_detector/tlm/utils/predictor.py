import torch
from torchvision.utils import draw_bounding_boxes
import numpy as np
import matplotlib.pyplot as plt


from peak_detector.tlm.model import TLModel
from peak_detector.tlm.utils.evaluator import TLMEvaluator
from peak_detector.sslm.utils.predictor import SSLPredictor


class TLMPredictor(SSLPredictor):
    def __init__(self, state_dict_path: str) -> None:
        self.state_dict_path = state_dict_path

        self.model = TLModel(2).model
        self.model.load_state_dict(
            torch.load(state_dict_path, map_location=torch.device("cpu"))
        )
        self.model.eval()
        self.height = 195
        self.width = 487

    @torch.no_grad()
    def predict(
        self, folder: str, nxs_no: str, img_no: int, threshold: float, use_nms=False
    ) -> tuple[list, np.ndarray, list]:
        """Generates the model predicted masks and boxes for the given scan from its nexus file

        Args:
            folder (str): folder containing the nexus file
            nxs_no (str): scan number of the specified nexus file
            img_no (int): image index within the scan
            threshold (float): confidence threshold to accept a prediction
            use_nms (bool, optional): whether to apply the nms algorithm to reduce overlap in predictions. Defaults to False.

        Returns:
            tuple[list, np.ndarray, list]: list of predicted masks, array of predicted boxes and list of scores for each
        """
        # load in image and get model prediction
        img_tensor = self.get_image(folder, nxs_no, img_no)
        pred_targets = self.model([img_tensor])[0]

        masks = pred_targets["masks"]
        out_masks = []
        labels = np.asarray(pred_targets["labels"])
        bboxes = np.asarray(pred_targets["boxes"])
        out_boxes = []
        scores = np.asarray(pred_targets["scores"])
        out_scores = []

        for count, score in enumerate(scores):
            if score > threshold:
                mask = np.asarray(masks[count][0])
                # 0.5 is the default threshold for object detection
                mask = np.where(mask > 0.5, 1, 0)
                out_masks.append(mask)
                out_boxes.append(bboxes[count])
                out_scores.append(scores[count])

        if use_nms:
            # dispose of some predictions which have too much overlap with more likely ones
            targets = {"boxes": out_boxes, "masks": out_masks, "scores": out_scores}
            eval = TLMEvaluator(
                None, self.state_dict_path, torch.device("cpu")  # type: ignore
            )
            targets = eval.non_max_supression(targets, 0.5)
            out_boxes = targets["boxes"]
            out_masks = targets["masks"]
            out_scores = targets["scores"]

        return out_masks, np.asarray(out_boxes), out_scores

    def show_output(
        self,
        folder: str,
        nxs_no: str,
        img_no: int,
        masks: list[np.ndarray],
        boxes: np.ndarray,
        show_boxes=False,
    ):
        """Display the predicted masks (and optionally boxes) on the original and log-scaled image

        Args:
            folder (str): folder containing the nexus file
            nxs_no (str): scan number of the nexus file
            img_no (int): image index within the scan data
            masks (list[np.ndarray]): list of masks to display over the picture
            boxes (np.ndarray): list of boxes to display over the picture
            show_boxes (bool, optional): whether to also display the bounding boxes. Defaults to False.
        """
        image = self.get_image(folder, nxs_no, img_no)
        allmasks = np.zeros((self.height, self.width), dtype=int)
        for mask in masks:
            allmasks = allmasks | mask

        if show_boxes and len(boxes) > 0:
            tboxes = torch.as_tensor(boxes, dtype=torch.int16)
            box_img = draw_bounding_boxes(image, tboxes)
            image = np.asarray(box_img)

        # re-arrange image data as pyplot expects [H, W, C] data
        image = np.transpose(image, [1, 2, 0])
        # add small offset to avoid taking log of 0
        log_image = np.log(image + 0.00001)

        fig = plt.figure()
        plt.suptitle("Transfer Learning Prediction")
        fig.add_subplot(2, 2, 1)
        plt.imshow(image, cmap="viridis")
        plt.imshow(allmasks, cmap="Greys", alpha=0.25)
        fig.add_subplot(2, 2, 2)
        plt.imshow(log_image, cmap="viridis")
        plt.imshow(allmasks, cmap="Greys", alpha=0.25)
        fig.add_subplot(2, 2, 3)
        plt.imshow(image, cmap="viridis")
        fig.add_subplot(2, 2, 4)
        plt.imshow(log_image, cmap="viridis")
        plt.show()
