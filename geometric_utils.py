import numpy as np
from scipy.ndimage import label


def threshold(x, threshold=None):
    """
    Function to take some array-like object
    and binary threshold it (turn all values at or
    above threshold to 1s, rest elements to 0)

    @param threshold    Threshold value (0<=threshold<=1)
    @param x            Array-like object to perform binary thresholding

    @returns            Thresholded object that is the same type as x
    """
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def true_positives(pred_matrix, gt_matrix, overlap=0.3):
    """
    Function to calculate the true positives
    Does so by determining at which elements the predictions and ground truths
    intersect, and then compares those points of intersection to ground truths
    to determine level of overlap 

    @param pred_matrix  Matrix of predictions (ASSUMED TO BE BINARY 1s and 0s)
    @param gt_matrix    Matrix of ground truths (ASSUMED TO BE BINARY 1s ans 0s)
    @param overlap      Proportion of overlap (between 0 and 1 inclusive) necessary to retain 
    overlapping islands

    @returns            Matrix of retained inersecting islands, number of retained islands
    """

    # Element wise multiplication of two binary arrays leaves 1s only where overlap is present
    intersection = pred_matrix * gt_matrix
    # Adding gt_matrix to intersection makes true positives 2 and false negatives 1
    intersection = intersection + gt_matrix

    """ When the intersection matrix is passed to label(), this structure matrix ensures 
    features touching diagonally are counted as a single feature """
    structure = [[1,1,1],
                 [1,1,1],
                 [1,1,1]]
    
    labeled_matrix, num_islands = label(intersection,structure=structure)
    
    """ Iterate over each island in the intersection matrix, discarding islands
    where overlap isn't >= threshold """
    retained_matrix = np.zeros_like(intersection)
    for i in range(num_islands):
        # Get the index of island number i+1 in labeled_matrix, corresponds to intersection too
        index = labeled_matrix == i+1
        max_value = np.max(intersection[index])
        island = intersection[index]





def f_score(pr, gt):
    """
    Calculates the F-Score of predictions with ground
    truths

    F1 = TP/TP+1/2(FP+FN)
    """

