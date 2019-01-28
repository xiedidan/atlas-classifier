import numpy as np
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()

        if alpha is None:
            self.alpha = torch.ones(num_classes, dtype=torch.float) / num_classes
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float)

        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, classifications, annotations):
        device = classifications.device

        if len(annotations.shape) == 1:
            # single-label
            P = nn.functional.softmax(classifications, dim=-1)

            class_mask = torch.zeros(classifications.shape, device=device)
            ids = annotations.view(-1, 1).detach()
            class_mask.scatter_(1, ids, 1.)

            alpha = torch.take(self.alpha, ids).to(device=device)

            probs = (P * class_mask).sum(1).reshape(-1, 1)
            log_p = probs.log()

            loss = -alpha * torch.pow((1 - probs), self.gamma) * log_p
        else:
            # multi-label
            # each label is treated independently
            P = torch.sigmoid(classifications)
            class_mask = annotations.detach()

            # alpha_factor = at
            alpha_factor = self.alpha.repeat(classifications.shape[0], 1).to(device=device)
            alpha_factor = torch.where(class_mask > 0, alpha_factor, 1 - alpha_factor)

            # probs = pt
            probs = torch.ones(class_mask.shape, device=device)
            probs = torch.where(class_mask > 0, P, 1 - P)
            log_p = probs.log()

            # (1 - pt) ^ gamma
            gamma_factor = torch.pow((1 - probs), self.gamma)

            # all threee parts' shapes are [N, C]
            loss = -alpha_factor * gamma_factor * log_p

            # final shape [N]
            loss = torch.sum(loss, dim=-1)

        if self.reduction is None:
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
