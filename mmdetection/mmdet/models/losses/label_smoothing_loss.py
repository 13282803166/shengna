import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import weight_reduce_loss
import random

def smooth_one_hot(true_labels: torch.Tensor, classes_count: int, classes, cate2id, smoothing_pair, smoothing=0.0):
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes_count))
    smooth_label = torch.empty(size=label_shape, device=true_labels.device)
    smooth_label.fill_(0.0)
    for index, item in enumerate(smooth_label):
        category_id = true_labels[index]
        if category_id == 0:
            continue

        category = classes[category_id - 1]
        if category in smoothing_pair:
            smoothing_labels = smoothing_pair[category]
            sub_confidence = smoothing / len(smoothing_labels)
            for smoothing_label in smoothing_labels:
                smoothing_id = cate2id[smoothing_label]
                smooth_label[index][smoothing_id] = sub_confidence

    #print(smooth_label)
    #k += 1
    #smooth_label.fill_(smoothing / (classes_count - 1))
    smooth_label.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return smooth_label

def label_smoothing_loss(pred, label, classes, cate2id, smoothing_pair, weight=None, reduction='mean', avg_factor=None):
    smoothing = 0.1 
    pred = pred.log_softmax(dim=-1)
    true_dist = smooth_one_hot(label, pred.shape[1], classes, cate2id, smoothing_pair, smoothing)
    loss = torch.mean(torch.sum(-true_dist * pred, dim=-1))
    return loss

def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    # element-wise losses
    loss = F.cross_entropy(pred, label, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()

    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


@LOSSES.register_module
class LabelSmoothLoss(nn.Module):

    def __init__(self,
                 classes=None,
                 smoothing_pair=None,
                 reduction='mean',
                 loss_weight=1.0):
        super(LabelSmoothLoss, self).__init__()
        self.classes = classes
        self.smoothing_pair = smoothing_pair
        self.reduction = reduction
        self.loss_weight = loss_weight
        if classes is not None:
            self.cat2label = {cat: i + 1 for i, cat in enumerate(classes)}

        self.cls_criterion = label_smoothing_loss

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.cls_criterion(cls_score, label, self.classes, self.cat2label, self.smoothing_pair, weight, reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss_cls

