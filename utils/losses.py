from torch import nn
from . import geometric_utils
import re

class DiceLoss(nn.Module):
    """
    Implementation of Dice Loss using custom implementation
    of F Score

    @param threshold (float)        Threshold for prediction binarization

    Dice Loss = 1 - F Score
    """

    def __init__(self,name=None):
        super().__init__()
        self._name = name
        self.activation = nn.Softmax2d()

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        else:
            return self._name

    def forward(self, prediction, target):
        return 1 - geometric_utils.dice_coeff(prediction, target)
