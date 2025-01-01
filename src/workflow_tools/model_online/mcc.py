import torch
from torch.nn.modules.loss import _Loss


class MCCLoss(_Loss):
    def __init__(self, eps: float = 1e-5):
        """Compute Matthews Correlation Coefficient Loss for image segmentation task.
        It only supports binary mode.

        Args:
            eps (float): Small epsilon to handle situations where all the samples in the dataset belong to one class

        Reference:
            https://github.com/kakumarabhishek/MCC-Loss
        """
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute MCC loss

        Args:
            y_pred (torch.Tensor): model prediction of shape (N, H, W) or (N, 1, H, W)
            y_true (torch.Tensor): ground truth labels of shape (N, H, W) or (N, 1, H, W)
            mask (torch.Tensor): binary mask (0 or 1) of shape (N, H, W) or (N, 1, H, W)

        Returns:
            torch.Tensor: loss value (1 - mcc)
        """
        if mask is None:
            mask = torch.ones_like(y_true)

        y_pred = torch.sigmoid(y_pred)

        y_true = y_true * mask
        y_pred = y_pred * mask

        bs = y_true.shape[0]
        y_true = y_true.view(bs, 1, -1)
        y_pred = y_pred.view(bs, 1, -1)
        mask = mask.view(bs, 1, -1)

        # Calculate all metrics only within masked regions
        tp = torch.sum(torch.mul(y_pred, y_true) * mask) + self.eps
        tn = torch.sum(torch.mul((1 - y_pred), (1 - y_true)) * mask) + self.eps
        fp = torch.sum(torch.mul(y_pred, (1 - y_true)) * mask) + self.eps
        fn = torch.sum(torch.mul((1 - y_pred), y_true) * mask) + self.eps

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(
            torch.add(tp, fp)
            * torch.add(tp, fn)
            * torch.add(tn, fp)
            * torch.add(tn, fn)
        )

        mcc = torch.div(numerator.sum(), denominator.sum() + 1.0)
        loss = 1.0 - mcc

        return loss
