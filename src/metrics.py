
import torch

def dice_score(pred, target, threshold=0.5, smooth=1):
    pred   = (torch.sigmoid(pred) > threshold).float()
    pred   = pred.view(-1)
    target = target.view(-1)
    inter  = (pred * target).sum()
    return (2*inter+smooth)/(pred.sum()+target.sum()+smooth)

def iou_score(pred, target, threshold=0.5, smooth=1):
    pred   = (torch.sigmoid(pred) > threshold).float()
    inter  = (pred * target).sum()
    union  = pred.sum() + target.sum() - inter
    return (inter+smooth)/(union+smooth)

def precision_recall(pred, target, threshold=0.5, smooth=1):
    pred = (torch.sigmoid(pred) > threshold).float()
    tp   = (pred * target).sum()
    fp   = (pred * (1-target)).sum()
    fn   = ((1-pred) * target).sum()
    prec = (tp+smooth)/(tp+fp+smooth)
    rec  = (tp+smooth)/(tp+fn+smooth)
    return prec, rec
