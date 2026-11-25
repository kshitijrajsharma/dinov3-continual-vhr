"""Unit tests for the weighted kNN forgetting-probe core, no external data needed.

Bank and query are always disjoint, matching real use (ImageNet train bank, val queries);
querying a set against itself would let each point match itself and mask the classifier.
"""

import torch
from dinov3_vhr.monitor.knn import weighted_knn_accuracy


def _clustered(num_classes: int, per_class: int, dim: int, spread: float, seed: int) -> tuple:
    """Gaussian clusters, one per class, centred on orthogonal unit axes."""
    generator = torch.Generator().manual_seed(seed)
    features, labels = [], []
    for cls in range(num_classes):
        centre = torch.zeros(dim)
        centre[cls] = 1.0
        features.append(centre + spread * torch.randn(per_class, dim, generator=generator))
        labels.append(torch.full((per_class,), cls))
    return torch.cat(features), torch.cat(labels)


def test_separable_clusters_are_perfectly_classified() -> None:
    bank_x, bank_y = _clustered(5, 40, 8, spread=0.05, seed=0)
    query_x, query_y = _clustered(5, 10, 8, spread=0.05, seed=1)
    accuracy = weighted_knn_accuracy(bank_x, bank_y, query_x, query_y, num_classes=5, k=10)
    assert accuracy == 1.0


def test_random_features_score_near_chance() -> None:
    generator = torch.Generator().manual_seed(1)
    bank_x = torch.randn(500, 16, generator=generator)
    bank_y = torch.randint(0, 4, (500,), generator=generator)
    query_x = torch.randn(200, 16, generator=generator)
    query_y = torch.randint(0, 4, (200,), generator=generator)
    accuracy = weighted_knn_accuracy(bank_x, bank_y, query_x, query_y, num_classes=4, k=20)
    assert 0.10 < accuracy < 0.40


def test_k_larger_than_bank_is_clamped() -> None:
    bank_x, bank_y = _clustered(3, 5, 4, spread=0.05, seed=0)
    query_x, query_y = _clustered(3, 5, 4, spread=0.05, seed=2)
    accuracy = weighted_knn_accuracy(bank_x, bank_y, query_x, query_y, num_classes=3, k=1000)
    assert accuracy == 1.0
