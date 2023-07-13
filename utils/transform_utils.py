import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F


class EnforceFloat():
    """
    Enforce Float transform for both mask and image
    """
    def __call__(self, 
                 img: np.ndarray,
                 mask: np.ndarray):

        assert isinstance(img, torch.Tensor)
        if mask is not None:
            assert isinstance(mask, torch.Tensor)
            return img.float(), mask.float()
        else:
            return img.float(), mask
        

class ToTensor():
    """
    To Tensor transform for both mask and image
    """
    def __call__(self,
                 img: np.ndarray,
                 mask: np.ndarray):

        img = F.to_tensor(img)
        if mask is not None:
            mask = F.to_tensor(mask)
            return img, mask
        else:
            return img, mask
        

class MultiCompose:
    """
    MultiCompose transforms for both mask and image
    """
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, img, mask):
        for tf in self.transforms:
            img, mask = tf(img, mask)
        
        return img, mask

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string