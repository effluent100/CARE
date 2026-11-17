"""Correction Acceptance via Risk Evaluation (CARE)."""

import torch
import torch.nn as nn

from .candidate import extract_gnp_score, text_guided_residual_fusion


def safe_minmax(values, eps=1e-8):
    dims = tuple(range(1, values.dim()))
    minimum = values.amin(dim=dims, keepdim=True)
    maximum = values.amax(dim=dims, keepdim=True)
    return (values - minimum) / (maximum - minimum + eps)


def map_entropy(values, eps=1e-8):
    values = safe_minmax(values).float()
    mass = values.flatten(1) + eps
    mass = mass / mass.sum(dim=1, keepdim=True)
    normalizer = torch.log(
        torch.tensor(float(mass.shape[1]), device=values.device)
    )
    return -(mass * torch.log(mass)).sum(dim=1) / normalizer


def map_dispersion(values, eps=1e-8):
    values = safe_minmax(values).float()
    batch_size, height, width = values.shape
    yy = torch.linspace(-1.0, 1.0, height, device=values.device)
    yy = yy.view(1, height, 1).expand(batch_size, height, width)
    xx = torch.linspace(-1.0, 1.0, width, device=values.device)
    xx = xx.view(1, 1, width).expand(batch_size, height, width)
    mass = values.sum(dim=(1, 2), keepdim=True) + eps
    center_y = (values * yy).sum(dim=(1, 2), keepdim=True) / mass
    center_x = (values * xx).sum(dim=(1, 2), keepdim=True) / mass
    variance = (
        values * ((yy - center_y) ** 2 + (xx - center_x) ** 2)
    ).sum(dim=(1, 2)) / mass.view(batch_size)
    return variance / 2.0


def build_gnp_batch(patch_features, image_size, rank=64, multiscale=True):
    maps = []
    with torch.no_grad():
        for batch_index in range(patch_features[0].shape[0]):
            sample_features = [
                feature[batch_index : batch_index + 1].detach()
                for feature in patch_features
            ]
            maps.append(
                extract_gnp_score(
                    sample_features,
                    image_size,
                    rank=rank,
                    multiscale=multiscale,
                )
            )
    return torch.stack(maps, dim=0).to(patch_features[0].device)


def build_candidate_batch(text_map, gnp_map, temperature=5.0):
    candidates = [
        text_guided_residual_fusion(
            gnp_map[index], text_map[index], temperature=temperature
        )
        for index in range(text_map.shape[0])
    ]
    return torch.stack(candidates, dim=0)


def risk_features(text_map, candidate_map, gnp_map):
    """Compute the ten-dimensional risk representation described in the paper."""
    text_normalized = safe_minmax(text_map)
    candidate_normalized = safe_minmax(candidate_map)
    gnp_normalized = safe_minmax(gnp_map)

    entropy_text = map_entropy(text_normalized)
    entropy_candidate = map_entropy(candidate_normalized)
    dispersion_text = map_dispersion(text_normalized)
    dispersion_candidate = map_dispersion(candidate_normalized)
    disagreement_text = torch.mean(
        torch.abs(text_normalized - gnp_normalized), dim=(1, 2)
    )
    disagreement_candidate = torch.mean(
        torch.abs(candidate_normalized - gnp_normalized), dim=(1, 2)
    )
    candidate_drift = torch.mean(
        torch.abs(candidate_normalized - text_normalized), dim=(1, 2)
    )
    risk_text = entropy_text + 0.5 * dispersion_text + 0.5 * disagreement_text
    risk_candidate = (
        entropy_candidate
        + 0.5 * dispersion_candidate
        + 0.5 * disagreement_candidate
        + 0.5 * candidate_drift
    )
    risk_difference = risk_text - risk_candidate

    return torch.stack(
        [
            risk_difference,
            entropy_text,
            entropy_candidate,
            dispersion_text,
            dispersion_candidate,
            disagreement_text,
            disagreement_candidate,
            candidate_drift,
            risk_text,
            risk_candidate,
        ],
        dim=1,
    )


class CorrectionAcceptancePolicy(nn.Module):
    """The 10-32-32-1 sample-wise acceptance policy used by CARE."""

    def __init__(self, input_dim=10, hidden_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return torch.sigmoid(self.network(features)).view(-1, 1, 1)


def fuse_with_acceptance(text_map, candidate_map, acceptance):
    return (1.0 - acceptance) * text_map + acceptance * candidate_map
