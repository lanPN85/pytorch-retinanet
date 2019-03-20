from retinanet import RetinaNet
from loss import FocalLoss


class UserModel(RetinaNet):
    num_anchors = 9

    def __init__(self):
        super().__init__(num_classes=2)

    def loss(output, target):
        loc_preds, cls_preds = output
        loc_targets, cls_targets = target
        return FocalLoss(self.num_classes)(loc_preds, loc_targets, cls_preds, cls_targets)
