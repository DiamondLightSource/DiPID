from torch import nn
from torch import Tensor
from torch.nn import functional as F

# using model architecture from: https://arxiv.org/pdf/2107.04934.pdf


class SSLModel(nn.Module):
    def __init__(self, num_classes: int, num_conv=2, greyscale=True) -> None:
        """Creates the CNN for self-supervised training

        Args:
            num_classes (int): number of classes the model can predict
            num_conv (int, optional): number of convolutional layers to add to the model. Defaults to 2.
            greyscale (bool, optional): whether the input image is greyscale or RGB. Defaults to True.
        """
        super().__init__()
        self.height = 487
        self.width = 195
        self.num_conv = num_conv
        if greyscale:
            self.in_channels = 1
        else:
            self.in_channels = 3
        # 100 channels is chosen arbitrarily - this is the max number of clusters which can be predicted
        self.out_channels = 100
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        for i in range(num_conv - 1):
            self.conv2.append(
                nn.Conv2d(
                    self.out_channels,
                    self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            self.norm2.append(nn.BatchNorm2d(self.out_channels))
        self.conv3 = nn.Conv2d(
            self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0
        )
        self.norm3 = nn.BatchNorm2d(self.out_channels)
        # prediction is made by taking the channel with the largest value in the channels dimension

    def forward(self, x: Tensor) -> Tensor:
        """pass the input through the layers of the network and generate a prediction

        Args:
            x (Tensor): input image to the network

        Returns:
            Tensor: predicted output with the specified number of channels
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.norm1(x)
        for i in range(self.num_conv - 1):
            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.norm2[i](x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.norm3(x)
        return x
