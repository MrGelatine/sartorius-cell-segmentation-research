import matplotlib
matplotlib.use('Qt5Agg')
from torchvision.transforms import functional as F
import torch.nn.functional
from pytorch_unet import *

class FocalCoef(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = nn.BCELoss()
        BCE = BCE(inputs,targets)

        return BCE

#Dice + BCE
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE = nn.BCELoss()

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        Dice_BCE = BCE(inputs,targets) + dice_loss

        return Dice_BCE

class IoUCoef(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoUCoef, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return IoU

class F_Score(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(F_Score, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = inputs.view(-1).detach().numpy()
        targets = targets.view(-1).detach().numpy()

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        total = inputs.size
        temp = np.logical_xor(np.asarray(targets),(np.asarray(inputs) > 0.70).astype(int).flatten())
        TPTF = (np.invert(np.asarray(temp))).astype(int).sum()
        Accuracy = (TPTF.sum()) / total
        Precision = TP/(TP+FP)
        Recall = TP / (TP+FN)
        F_score = 2*(Precision*Recall)/(Precision+Recall)

        return (Accuracy,Precision,Recall,F_score)

ALPHA = 10.0
BETA = 0.7
class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

        return 1 - Tversky

class CCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.critetion = nn.CrossEntropyLoss()
    def forward(self, input, target):

        input = torch.softmax(input,dim=1)
        loss = self.critetion(input,target)

        return loss