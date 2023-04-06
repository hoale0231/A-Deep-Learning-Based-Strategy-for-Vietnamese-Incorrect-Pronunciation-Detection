# MIT License
#
# Copyright (c) 2021 Soohwan Kim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor


class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean') -> None:
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction.lower()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=self.reduction)
        
    def forward(
            self,
            decoder_log_probs: Tensor,
            targets: Tensor,
    ) -> Tensor:
        cross_entropy_loss = self.cross_entropy_loss(decoder_log_probs, targets.contiguous().view(-1))
        return cross_entropy_loss
    

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class JointCTCCrossEntropyLoss(nn.Module):
    """
    Privides Joint CTC-CrossEntropy Loss function

    Args:
        num_classes (int): the number of classification
        ignore_index (int): indexes that are ignored when calculating loss
        dim (int): dimension of calculation loss
        reduction (str): reduction method [sum, mean] (default: mean)
        ctc_weight (float): weight of ctc loss
        cross_entropy_weight (float): weight of cross entropy loss
        blank_id (int): identification of blank for ctc
    """
    def __init__(
            self,
            ignore_index: int,
            reduction='mean',
            ctc_weight: float = 0.3,
            cross_entropy_weight: float = 0.7,
            blank_id: int = None,
    ) -> None:
        super(JointCTCCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction.lower()
        self.ctc_weight = ctc_weight
        self.cross_entropy_weight = cross_entropy_weight
        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction=self.reduction, zero_infinity=True)
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=self.reduction, ignore_index=self.ignore_index)

    def forward(
            self,
            encoder_log_probs: Tensor,
            decoder_log_probs: Tensor,
            output_lengths: Tensor,
            targets: Tensor,
            target_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        ctc_loss = self.ctc_loss(encoder_log_probs, targets, output_lengths, target_lengths) if self.ctc_weight > 0 else 0
        cross_entropy_loss = self.cross_entropy_loss(decoder_log_probs, targets.contiguous().view(-1)) if self.cross_entropy_weight > 0 else 0
        loss = cross_entropy_loss * self.cross_entropy_weight + ctc_loss * self.ctc_weight
        return loss

class JointLoss(nn.Module):
    def __init__(
            self,
            ignore_index: int,
            reduction='mean',
            blank_id: int = None,
            ctc_weight: float = 0.3,
            cross_entropy_weight: float = 0.7,
            md_weight: float = 0.7,
            pr_weight: float = 0.3,
            gamma: float = 1.0
    ) -> None:
        super(JointLoss, self).__init__()
        self.pr_loss = JointCTCCrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction,
            blank_id=blank_id,
            cross_entropy_weight=cross_entropy_weight,
            ctc_weight=ctc_weight,
        )
        self.md_loss = FocalLoss(reduction=reduction, gamma=gamma)
        # self.md_loss = CrossEntropyLoss(reduction=reduction)
        self.md_weight = md_weight
        self.pr_weight = pr_weight

    def forward(
            self,
            encoder_log_probs: Tensor,
            pr_log_probs: Tensor,
            encoder_output_lengths: Tensor,
            r_os: Tensor,
            r_os_lengths: Tensor,
            md_log_probs: Tensor,
            score: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        md_loss = self.md_loss(md_log_probs, score) if self.md_weight > 0 else 0
        pr_loss = self.pr_loss(
            encoder_log_probs, pr_log_probs, 
            encoder_output_lengths, r_os, r_os_lengths
        ) if self.pr_weight > 0 and len(r_os) != 0 else 0
        loss = md_loss * self.md_weight + pr_loss * self.pr_weight
        return loss, pr_loss, md_loss

class WeaklyLoss(nn.Module):
    def __init__(
            self,
            ignore_index: int,
            reduction='mean',
            md_weight: float = 0.7,
            pr_weight: float = 0.3,
            gamma: float = 1.0
    ) -> None:
        super(WeaklyLoss, self).__init__()
        self.pr_loss = nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index)
        self.md_loss = FocalLoss(reduction=reduction, gamma=gamma)
        # self.md_loss = CrossEntropyLoss(reduction=reduction)
        self.md_weight = md_weight
        self.pr_weight = pr_weight

    def forward(
            self,
            pr_log_probs: Tensor,
            r_os: Tensor,
            md_log_probs: Tensor,
            score: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        md_loss = self.md_loss(md_log_probs, score) if self.md_weight > 0 else 0
        pr_loss = self.pr_loss(pr_log_probs, r_os.contiguous().view(-1)) if self.pr_weight > 0 and len(r_os) != 0 else 0
        loss = md_loss * self.md_weight + pr_loss * self.pr_weight
        return loss, pr_loss, md_loss
