
import os
import numpy as np
import torch
import SimpleITK as sitk
from monai.utils import convert_to_tensor
from datetime import datetime
from collections import OrderedDict

# from monai.networks import one_hot
def saveToNii(out_file, img_mask, sitkcast=False, rot180=False, out_dir=None):
    print(out_file+" img_mask.shape", img_mask.shape)
    if out_dir==None:
        out_dir="/data3/saveToNii/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if img_mask.shape[0] != 1:
        img_mask = torch.argmax(img_mask, 0)
    else:
        img_mask = img_mask[0, :]
    ## save
    # print("1 img_mask.shape", img_mask.shape)
    # img_mask = np.transpose(img_mask, (0, 1, 2))
    if rot180:
        img_mask = np.rot90(img_mask, axes=(1, 2))
        img_mask = np.rot90(img_mask, axes=(1, 2))
    img = sitk.GetImageFromArray(img_mask)
    if sitkcast:
        img = sitk.Cast(img, sitk.sitkUInt8)  # !!!!change 5 to 1,need cast sitkInt16 to sitkInt8
    sitk.WriteImage(img, os.path.join(out_dir, out_file))
    print("save to ",out_dir,out_file)

from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
from monai.transforms.utils import generate_spatial_bounding_box, compute_divisible_spatial_size
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.utils_pytorch_numpy_unification import floor_divide, maximum
def compute_bounding_box(img: NdarrayOrTensor, k_divisible: Union[Sequence[int], int] = 1):
        """
        Compute the start points and end points of bounding box to crop.
        And adjust bounding box coords to be divisible by `k`.

        """
        box_start, box_end = generate_spatial_bounding_box(img=img)
        box_start_, *_ = convert_data_type(box_start, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        box_end_, *_ = convert_data_type(box_end, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        orig_spatial_size = box_end_ - box_start_
        # make the spatial size divisible by `k`
        spatial_size = np.asarray(compute_divisible_spatial_size(orig_spatial_size.tolist(), k=k_divisible))
        # update box_start and box_end
        box_start_ = box_start_ - np.floor_divide(np.asarray(spatial_size) - orig_spatial_size, 2)
        box_end_ = box_start_ + spatial_size
        return box_start_, box_end_


def SpatialCrop(img: NdarrayOrTensor,
        roi_center: Union[Sequence[int], NdarrayOrTensor, None] = None,
        roi_size: Union[Sequence[int], NdarrayOrTensor, None] = None,
        roi_start: Union[Sequence[int], NdarrayOrTensor, None] = None,
        roi_end: Union[Sequence[int], NdarrayOrTensor, None] = None,
        roi_slices: Optional[Sequence[slice]] = None,):
    """
    General purpose cropper to produce sub-volume region of interest (ROI).
    If a dimension of the expected ROI size is bigger than the input image size, will not crop that dimension.
    So the cropped result may be smaller than the expected ROI, and the cropped results of several images may
    not have exactly the same shape.
    It can support to crop ND spatial (channel-first) data.

    The cropped region can be parameterised in various ways:
        - a list of slices for each spatial dimension (allows for use of -ve indexing and `None`)
        - a spatial center and size
        - the start and end coordinates of the ROI

    Args:
            roi_center: voxel coordinates for center of the crop ROI.
            roi_size: size of the crop ROI, if a dimension of ROI size is bigger than image size,
                will not crop that dimension of the image.
            roi_start: voxel coordinates for start of the crop ROI.
            roi_end: voxel coordinates for end of the crop ROI, if a coordinate is out of image,
                use the end coordinate of image.
            roi_slices: list of slices for each of the spatial dimensions.
        """
    roi_start_torch: torch.Tensor

    if roi_slices:
            if not all(s.step is None or s.step == 1 for s in roi_slices):
                raise ValueError("Only slice steps of 1/None are currently supported")
            slices = list(roi_slices)
    else:
            if roi_center is not None and roi_size is not None:
                roi_center, *_ = convert_data_type(
                    data=roi_center, output_type=torch.Tensor, dtype=torch.int16, wrap_sequence=True
                )
                roi_size, *_ = convert_to_dst_type(src=roi_size, dst=roi_center, wrap_sequence=True)
                _zeros = torch.zeros_like(roi_center)
                roi_start_torch = maximum(roi_center - floor_divide(roi_size, 2), _zeros)  # type: ignore
                roi_end_torch = maximum(roi_start_torch + roi_size, roi_start_torch)
            else:
                if roi_start is None or roi_end is None:
                    raise ValueError("Please specify either roi_center, roi_size or roi_start, roi_end.")
                roi_start_torch, *_ = convert_data_type(
                    data=roi_start, output_type=torch.Tensor, dtype=torch.int16, wrap_sequence=True
                )
                roi_start_torch = maximum(roi_start_torch, torch.zeros_like(roi_start_torch))  # type: ignore
                roi_end_torch, *_ = convert_to_dst_type(src=roi_end, dst=roi_start_torch, wrap_sequence=True)
                roi_end_torch = maximum(roi_end_torch, roi_start_torch)
            # convert to slices (accounting for 1d)
            if roi_start_torch.numel() == 1:
                slices = [slice(int(roi_start_torch.item()), int(roi_end_torch.item()))]
            else:
                slices = [slice(int(s), int(e)) for s, e in zip(roi_start_torch.tolist(), roi_end_torch.tolist())]
    #crop
    sd = min(len(slices), len(img.shape[1:]))  # spatial dims
    slices = [slice(None)] + slices[:sd]
    return img[tuple(slices)]