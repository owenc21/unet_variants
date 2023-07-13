from torch import nn
from geometric_utils import f_score

class DiceLoss(nn.Module):
    """
    Implementation of Dice Loss using custom implementation
    of F Score

    @param threshold (float)        Threshold for prediction binarization

    Dice Loss = 1 - F Score
    """

    def __init__(self, threshold):
        super(DiceLoss, self).__init__()
        self.threshold = threshold

    def forward(self, prediction, target):
        return 1 - f_score(prediction, target, self.threshold)

    def __call__(self, prediction, target):
        return 1 - f_score(prediction, target, self.threshold)