# -*- coding: utf-8 -*-
# Standard library imports
import sys

import torch


def rule(scores, labels, similarity_threshold=0.45):
    scores = scores.squeeze(0)
    approx_tops = torch.argsort(scores, descending=True)
    top_index = approx_tops.data.cpu().tolist()[0]
    if scores[top_index] >= similarity_threshold:
        return [labels[top_index], scores[top_index].cpu().detach().numpy().item()], None, [top_index, -1]
    else:
        return ["Unknown", scores[top_index].cpu().detach().numpy().item()], None, [top_index, -1]
