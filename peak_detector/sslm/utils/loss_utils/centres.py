import numpy as np
import torch
from torch import Tensor

# taken directly from https://github.com/osmond332/Spatial_Guided_Self_Supervised_Clustering/blob/main/src/center.py


def get_coordinate_tensors(
    x_max: int, y_max: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Create flat tensors of the correct scale and dimension to calculate centres and variances

    Args:
        x_max (int): maximum value in the x direction
        y_max (int): maximum value in the y direction
        device (torch.device): set to GPU when tensors need to be loaded to GPU, CPU otherwise

    Returns:
        tuple[Tensor, Tensor]: scaled and flattened tensors for x and y calculations
    """
    x_map = np.tile(np.arange(x_max), (y_max, 1)) / x_max * 2 - 1.0
    y_map = np.tile(np.arange(y_max), (x_max, 1)).T / y_max * 2 - 1.0

    x_map_tensor = torch.from_numpy(x_map.astype(np.float32)).to(device)
    y_map_tensor = torch.from_numpy(y_map.astype(np.float32)).to(device)

    return x_map_tensor, y_map_tensor


def get_centre(
    part_map: Tensor, device: torch.device, self_referenced=False
) -> tuple[Tensor, Tensor]:
    """Finds the centre of all non-zero pixels in the input tensor

    Args:
        part_map (Tensor): binary map
        device (torch.device): set to GPU when tensors need to be loaded to GPU, CPU otherwise
        self_referenced (bool, optional): whether to clone the tensors before returning. Defaults to False.

    Returns:
        tuple[Tensor, Tensor]: tensors with the x and y co-ordinates of the centre
    """
    h, w = part_map.shape
    x_map, y_map = get_coordinate_tensors(h, w, device)
    x_map = torch.transpose(x_map, 1, 0)
    y_map = torch.transpose(y_map, 1, 0)

    x_center = (part_map * x_map).sum()
    y_center = (part_map * y_map).sum()

    if self_referenced:
        x_c_value = float(x_center.cpu().detach())
        y_c_value = float(y_center.cpu().detach())
        x_center = (part_map * (x_map - x_c_value)).sum() + x_c_value
        y_center = (part_map * (y_map - y_c_value)).sum() + y_c_value

    return x_center, y_center


def get_centres(
    part_maps: Tensor,
    device: torch.device,
    channels: int,
    epsilon=1e-3,
    self_ref_coord=False,
) -> Tensor:
    """Find centre of each channel in a predicted output

    Args:
        part_maps (Tensor): each channel has a 2d tensor binary map
        device (torch.device): set to GPU when tensors need to be loaded to GPU, CPU otherwise
        channels (int): number of channels in the input
        epsilon (_type_, optional): small offset to avoid 0 errors. Defaults to 1e-3.
        self_ref_coord (bool, optional): whether to clone co-ordinate tensors before returning. Defaults to False.

    Returns:
        Tensor: _description_
    """
    centers = []
    for c in range(channels):
        part_map = part_maps[c, :, :] + epsilon
        k = part_map.sum()
        part_map_pdf = part_map / k
        x_c, y_c = get_centre(part_map_pdf, device, self_ref_coord)
        centers.append(torch.stack((x_c, y_c), dim=0).unsqueeze(0))
    return torch.cat(centers, dim=0)
