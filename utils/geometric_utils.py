import numpy as np
from scipy.ndimage import label
import torch
from torch import Tensor


def _threshold(x, threshold=None):
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


def tp_fn(pred_matrix, gt_matrix, overlap=0.3):
    """
    Function to calculate the true positives and false negatives
    Does so by determining at which elements the predictions and ground truths
    intersect, and then compares those points of intersection to ground truths
    to determine level of overlap 

    If island has overlap >= overlap, it is a true positive
    If island is all 1s or overlap < overlap, it is a false negative

    @param pred_matrix  Matrix of predictions (ASSUMED TO BE BINARY 1s and 0s)
    @param gt_matrix    Matrix of ground truths (ASSUMED TO BE BINARY 1s ans 0s)
    @param overlap      Proportion of overlap (between 0 and 1 inclusive) necessary to retain 
    overlapping islands

    @returns            Matrix of retained inersecting islands,
                        Matrix of islands of false negatives,
                        number of retained islands (tp), false negatives (fn)
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
    false_negatives = 0
    fn_matrix = np.zeros_like(intersection)
    for i in range(num_islands):
        # Get the index of island number i+1 in labeled_matrix, corresponds to intersection too
        index = labeled_matrix == i+1
        max_value = np.max(intersection[index])
        island = intersection[index]
        # Count number of elements in intersection matrix island that are 1 and 2 respectively
        island_ones = (island[island==1].shape[0])
        island_twos = (island[island==2].shape[0])
        total_area = island_ones + island_twos
        if total_area == 0:
            pass
        overlap_actual = island_twos / total_area
        # If proportion of overlap is greater than overlap treshold, retain that island
        if overlap_actual >= overlap:
            retained_matrix[island] = 1
        # Else, this island is all 1s or doesn't have necessary overlap -> is a false negative
        else:
            false_negatives += 1
            fn_matrix[island] = 1


    _, num_retained_islands = label(retained_matrix, structure=structure)

    return retained_matrix, fn_matrix, num_retained_islands, false_negatives


def recall(pr, gt, threshold=0.3, overlap=0.3):
    """
    Calculates the Recall of predictions with ground
    truths

    Recall = TP/(TP+FN)

    @param pr           Prediction matrix (tensor)
    @param gt           Ground truth matrix (tensor)
    @param threshold    Threshold for pr binarization
    @param overlap      Proportion of overlap between prediction and gt
    necessary for it to be counted as true positive

    @returns        True positives, Ground truth count, recall (float)
    """

    batch_size = pr.shape[0]
    num_channels = pr.shape[1]

    tp = 0
    total = 0

    for b in range(batch_size):
        for c in range(num_channels):
            # Threshold pr matrix and set all values in gt matrix between 0 and 1 (exlcusive) to 1
            pr_matrix = _threshold(pr, threshold)
            gt_matrix = gt
            gt_matrix[(gt_matrix>0)&(gt_matrix<1)] = 1

            pr_matrix = pr_matrix.cpu().numpy()
            gt_matrix = gt_matrix.cpu().numpy()

            tp_matrix, tp_count, fn_matrix, fn_count = tp_fn(
                pred_matrix=pr_matrix, gt_matrix=gt_matrix
            )

            tp += tp_count
            total += tp_count + fn_count


    # Avoid divide by 0
    if total == 0:
        return torch.nan
    
    return tp, total, tp/total


def iou(pr, gt, threshold=0.3):
    """
    Calculates the intersection over union (Jaccard score) between
    ground truth and prediction

    @param pr           Prediction matrix (tensor)
    @param gt           Ground truth matrix (tensor)
    @param threshold    Threshold for pr binarization

    @returns            IoU (Jaccard score)
    """

    batch_size = pr.shape[0]
    num_channels = pr.shape[1]

    iou = 0
    iou_N = 0
    for b in range(batch_size):
        for c in range(num_channels):
            # Threshold pr matrix and set all values in gt matrix between 0 and 1 (exlcusive) to 1
            pr_matrix = pr[b,c,:,:]
            pr_matrix = _threshold(pr_matrix, threshold)
            gt_matrix = gt[b,c,:,:]
            gt_matrix[(gt_matrix>0)&(gt_matrix<1)] = 1

            intersection = torch.sum(gt_matrix*pr_matrix) # Count elements of 1
            union = torch.sum(pr_matrix+gt_matrix - intersection) # Count elements of 1

            iou_N += 1
            if union==0:
                pass

            iou += intersection / union

    if iou_N == 0:
        return 0
    
    return iou/iou_N


def dice_coeff(input: Tensor, target: Tensor, epsilon: float = 1e-6):
    """
    Function to determine dice coefficient between input and target

    @param input        The input tensor to determine dice score of
    @param target       Target (ground truth) tensor used to determine dice score

    @returns            Dice Coefficient/Score (float)
    """
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3

    sum_dim = (-1, -2)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()
