"""Weighted kNN accuracy, the forgetting-probe core.

Matches the DINO evaluation protocol also used by the pangaea `cls_knn` decoder:
cosine similarity, temperature-weighted votes, k=200, t=0.1. Extract frozen class-token
features for a labelled bank and a query split, then read top-1 accuracy. A drop on the
source domain (ImageNet) against the frozen baseline is forgetting.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def weighted_knn_accuracy(
    bank_features: Tensor,
    bank_labels: Tensor,
    query_features: Tensor,
    query_labels: Tensor,
    num_classes: int,
    k: int = 200,
    temperature: float = 0.1,
) -> float:
    """Top-1 accuracy of a temperature-weighted cosine kNN classifier.

    bank_* are the labelled reference set, query_* the evaluation set. Votes from the k
    nearest bank neighbours are weighted by exp(cosine_similarity / temperature).
    """
    if bank_features.ndim != 2 or query_features.ndim != 2:
        raise ValueError("features must be 2D (N, dim)")
    neighbours = min(k, bank_features.shape[0])

    bank = F.normalize(bank_features.float(), dim=1)
    query = F.normalize(query_features.float(), dim=1)
    similarity = query @ bank.t()

    top_similarity, top_index = similarity.topk(neighbours, dim=1)
    top_labels = bank_labels[top_index]
    weights = (top_similarity / temperature).exp()

    votes = torch.zeros(query.shape[0], num_classes, device=query.device)
    votes.scatter_add_(1, top_labels, weights)
    predictions = votes.argmax(dim=1)
    return (predictions == query_labels).float().mean().item()
