import torchvision
import torch
import torch.nn as nn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class TLModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = self.make_model(num_classes)

    def make_model(self, num_classes: int):
        """Builds the transfer model by adding a new head onto the pre-trained ResNet backbone

        Args:
            num_classes (int): Number of classes the model is able to predict. 0 is reserved for background but IS included in the count

        Returns:
            model: The model generated with initial weights frozen and a new head ready for training
        """
        # load pre-trained model and fix weights
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        model.requires_grad_(False)

        # add new fastrcnn head for bounding box prediction
        in_features = model.roi_heads.box_predictor.cls_score.in_features  # type: ignore
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.roi_heads.box_predictor.requires_grad_(True)

        # add new maskrcnn head for mask prediction
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels  # type: ignore
        hidden_layer = 256  # this value can be changed to change the size of the model
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
        model.roi_heads.mask_predictor.requires_grad_(True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # if possible, load the model to be capable of running across multiple GPUs
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
        model.to(device)

        return model

    def check_frozen_parameters(self):
        """Tests that the model weights correctly freeze and unfreeze"""
        frozen = 0
        unfrozen = 0
        for param in self.model.parameters():
            for element in param:
                if element.requires_grad == False:
                    frozen += 1
                elif element.requires_grad == True:
                    unfrozen += 1
        print(f"frozen: {frozen}; unfrozen: {unfrozen}")


if __name__ == "__main__":
    model = TLModel(2).make_model(2)
