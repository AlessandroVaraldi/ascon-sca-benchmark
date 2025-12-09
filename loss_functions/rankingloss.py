import torch
import torch.nn as nn
import torch.nn.functional as F 

class RankingLoss(nn.Module):
    """
    Pairwise Ranking Loss (RkL) as in Zaid et al. (TCHES 2021).

    For each sample i with logits s (shape [C]) and true class y:

        L_i = sum_{k != y} log(1 + exp(-alpha * (s_y - s_k)))

    Here we usually have C=2 (binary bit 0/1), but implementation is generic.
    """

    def __init__(self, alpha: float = 5.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Tensor of shape [B, C]
            targets: LongTensor of shape [B], each in [0, C-1]
        """
        B, C = logits.shape

        # s_pos: [B, 1]
        s_pos = logits.gather(1, targets.view(-1, 1))
        s_pos_expanded = s_pos.expand(-1, C)  # [B, C]

        diff = s_pos_expanded - logits  # [B, C]

        # Mask out correct class
        mask = torch.ones_like(diff, dtype=torch.bool)
        mask.scatter_(1, targets.view(-1, 1), False)

        diff_neg = diff[mask]  # [B*(C-1)]

        # log(1 + exp(-alpha * diff))
        loss_per_pair = F.softplus(-self.alpha * diff_neg)
        return loss_per_pair.mean()