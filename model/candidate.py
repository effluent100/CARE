"""Structural candidate construction used by CARE."""

import torch
import torch.nn.functional as F


def low_rank_residual_scores(features, rank=64):
    """Return cosine reconstruction residuals from a truncated low-rank model."""
    mean = features.mean(dim=0, keepdim=True)
    centered = features - mean
    actual_rank = min(rank, min(centered.shape))

    u, s, v = torch.svd_lowrank(centered, q=actual_rank)
    reconstruction = mean + u @ torch.diag(s) @ v.T

    scores = 1.0 - F.cosine_similarity(features, reconstruction, dim=-1)
    return F.relu(scores - scores.mean())


def extract_gnp_score(patch_features, image_size, rank=64, multiscale=True):
    """Construct the Global Normality Prototype (GNP) residual map.

    Args:
        patch_features: Feature tensors from selected vision-transformer layers.
        image_size: Spatial size of the output anomaly map.
        rank: Base low-rank dimension.
        multiscale: Use layer-wise rank multipliers [8, 4, 2, 1].
    """
    if isinstance(rank, (list, tuple)):
        ranks = list(rank)
    elif multiscale:
        ranks = [rank * 8, rank * 4, rank * 2, rank]
    else:
        ranks = rank

    maps = []
    for index, patch_feature in enumerate(patch_features):
        # The first token is the class token and is not part of the patch grid.
        patches = patch_feature.squeeze(0)[1:, :]
        side = int(patches.shape[0] ** 0.5)
        patches = F.normalize(patches, dim=-1)
        layer_rank = ranks[index] if isinstance(ranks, list) else ranks
        score = low_rank_residual_scores(patches, rank=layer_rank)
        score_map = score.view(1, 1, side, side)
        score_map = F.interpolate(
            score_map,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        maps.append(score_map)

    gnp_map = torch.stack(maps, dim=0).mean(dim=0)
    minimum = gnp_map.amin()
    maximum = gnp_map.amax()
    gnp_map = (gnp_map - minimum) / (maximum - minimum + 1e-8)
    return gnp_map.squeeze(0).squeeze(0)


def text_guided_residual_fusion(gnp_map, text_map, temperature=5.0):
    """Build a semantically constrained candidate with text-guided fusion."""
    minimum = text_map.amin()
    maximum = text_map.amax()
    normalized_text = (text_map - minimum) / (maximum - minimum + 1e-8)
    semantic_gate = torch.sigmoid(
        (normalized_text - normalized_text.mean()) * temperature
    )
    return semantic_gate * text_map + (1.0 - semantic_gate) * gnp_map
