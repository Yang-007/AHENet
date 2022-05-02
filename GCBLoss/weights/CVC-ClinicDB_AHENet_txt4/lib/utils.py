import pdb
import random
import cv2
import numpy as np



def calc_metric(pre, gt):
    epsilon = 1e-20
    b = pre.size(0)
    pre = pre.view(b, -1)
    gt = gt.view(b, -1)
    
    tp = ((pre == 1).float() * (gt == 1).float()).sum(-1)
    fn = ((pre == 0).float() * (gt == 1).float()).sum(-1)
    fp = ((pre == 1).float() * (gt == 0).float()).sum(-1)

    recall = (tp + epsilon) / (tp + fn + epsilon)
    precision = (tp + epsilon) / (tp + fp + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall)
    iou_score = (tp + epsilon) / (fp + tp + fn + epsilon)
    #dice = 2 * tp / ((pre == 1).float().sum(-1)+(gt == 1).float().sum(-1) + epsilon)
    return f1_score, iou_score


def gaussian_blur(img, max_ksize=3):
    """Apply Gaussian blur to input images."""
    ksize = np.random.randint(0, max_ksize, size=(2,))
    ksize = tuple((ksize * 2 + 1).tolist())

    img = cv2.GaussianBlur(img, ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_REPLICATE)
    return img


def median_blur(img, max_ksize=3):
    """Apply median blur to input images."""
    ksize = np.random.randint(0, max_ksize)
    ksize = ksize * 2 + 1
    img = cv2.medianBlur(img, ksize)
    return img

