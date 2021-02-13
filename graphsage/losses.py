import torch


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=0.5, reduction="mean"):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.sum_reduction = reduction == "sum"

    def forward(self, anchor, positive, negative):
        pos = (anchor - positive).pow(2).sum(1)
        neg = (anchor - negative).pow(2).sum(1)
        losses = torch.nn.functional.relu(pos - neg + self.margin)
        if self.sum_reduction:
            return losses.sum()
        return losses.mean()
