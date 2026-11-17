"""Train the source-supervised correction-acceptance policy used by CARE."""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataloader.dataset import Dataset
from model.backbone import compute_similarity, get_similarity_map, load_backbone
from model.care import (
    CorrectionAcceptancePolicy,
    build_candidate_batch,
    build_gnp_batch,
    fuse_with_acceptance,
    risk_features,
    safe_minmax,
)
from model.prompt_ensemble import AnomalyCLIP_PromptLearner
from utils.logger import get_logger
from utils.losses import BinaryDiceLoss
from utils.transforms import get_transform


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def per_sample_bce(prediction, target):
    prediction = prediction.clamp(1e-6, 1.0 - 1e-6)
    loss = F.binary_cross_entropy(prediction, target, reduction="none")
    return loss.flatten(1).mean(dim=1)


def build_text_map(model, image, prompt_learner, args):
    _, patch_features = model.encode_image(
        image, args.features_list, DPAM_layer=20
    )
    prompts, tokenized_prompts, compound_prompts = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(
        prompts, tokenized_prompts, compound_prompts
    ).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    anomaly_maps = []
    selected_patch_features = []
    for index, patch_feature in enumerate(patch_features):
        if index < args.feature_map_layer[0]:
            continue
        selected_patch_features.append(patch_feature)
        normalized_patch = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
        similarity, _ = compute_similarity(
            normalized_patch, text_features[0]
        )
        similarity_map = get_similarity_map(
            similarity[:, 1:, :], args.image_size
        ).permute(0, 3, 1, 2)
        anomaly_maps.append(
            (similarity_map[:, 1, :, :] + 1.0 - similarity_map[:, 0, :, :])
            / 2.0
        )

    return torch.stack(anomaly_maps, dim=0).sum(dim=0), selected_patch_features


def train(args):
    os.makedirs(args.save_path, exist_ok=True)
    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess, target_transform = get_transform(args)

    prompt_config = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
    }
    model, _ = load_backbone(
        "ViT-L/14@336px", device=device, design_details=prompt_config
    )
    model.visual.DAPM_replace(DPAM_layer=20)
    model.eval()

    dataset = Dataset(
        root=args.train_data_path,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=args.dataset,
        mode=args.train_split,
        meta_file=args.meta_file or None,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if not os.path.isfile(args.prompt_checkpoint):
        raise FileNotFoundError(args.prompt_checkpoint)
    prompt_checkpoint = torch.load(args.prompt_checkpoint, map_location="cpu")
    prompt_learner = AnomalyCLIP_PromptLearner(
        model.to("cpu"), prompt_config
    ).to(device)
    prompt_learner.load_state_dict(prompt_checkpoint["prompt_learner"], strict=True)
    prompt_learner.eval()
    for parameter in prompt_learner.parameters():
        parameter.requires_grad_(False)

    model.to(device)
    policy = CorrectionAcceptancePolicy(hidden_dim=args.policy_hidden_dim).to(device)
    policy.train()
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=args.learning_rate,
        betas=(0.5, 0.999),
        weight_decay=0.0,
    )
    dice_loss = BinaryDiceLoss()

    for key, value in sorted(vars(args).items()):
        logger.info("%s = %s", key, value)
    logger.info(
        "trainable policy parameters = %d",
        sum(parameter.numel() for parameter in policy.parameters()),
    )

    for epoch in range(args.epochs):
        losses = []
        acceptance_values = []
        target_accuracy = []

        for step, batch in enumerate(tqdm(loader, desc=f"epoch {epoch + 1}")):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break

            image = batch["img"].to(device)
            target = batch["img_mask"].to(device)
            if target.dim() == 3:
                target = target.unsqueeze(1)
            target = (target.float() > 0.5).float().squeeze(1)

            with torch.no_grad():
                text_map, patch_features = build_text_map(
                    model, image, prompt_learner, args
                )
                gnp_map = build_gnp_batch(
                    patch_features,
                    args.image_size,
                    rank=args.gnp_rank,
                    multiscale=True,
                )
                candidate_map = build_candidate_batch(
                    text_map, gnp_map, temperature=args.tgrf_temperature
                )
                features = risk_features(text_map, candidate_map, gnp_map)
                text_normalized = safe_minmax(text_map)
                candidate_normalized = safe_minmax(candidate_map)
                gnp_normalized = safe_minmax(gnp_map)
                target_acceptance = (
                    per_sample_bce(candidate_normalized, target)
                    < per_sample_bce(text_normalized, target)
                ).float().view(-1, 1, 1)

            acceptance = policy(features)
            final_map = fuse_with_acceptance(text_map, candidate_map, acceptance)
            final_normalized = safe_minmax(final_map)

            localization_loss = F.binary_cross_entropy(
                final_normalized.clamp(1e-6, 1.0 - 1e-6), target
            ) + dice_loss(final_normalized, target)
            acceptance_loss = F.binary_cross_entropy(
                acceptance.clamp(1e-6, 1.0 - 1e-6), target_acceptance
            )
            normal_suppression = (
                acceptance * (1.0 - target) * gnp_normalized
            ).mean()
            total_loss = (
                localization_loss
                + args.acceptance_loss_weight * acceptance_loss
                + args.suppression_weight * normal_suppression
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            losses.append(float(total_loss.detach().cpu()))
            acceptance_values.append(float(acceptance.mean().detach().cpu()))
            target_accuracy.append(
                float(
                    (
                        (acceptance.view(-1) > 0.5)
                        == (target_acceptance.view(-1) > 0.5)
                    )
                    .float()
                    .mean()
                    .detach()
                    .cpu()
                )
            )

        logger.info(
            "epoch [%d/%d], loss=%.4f, acceptance=%.4f, target_acc=%.4f",
            epoch + 1,
            args.epochs,
            float(np.mean(losses)),
            float(np.mean(acceptance_values)),
            float(np.mean(target_accuracy)),
        )

        checkpoint_path = os.path.join(args.save_path, f"epoch_{epoch + 1}.pth")
        torch.save(
            {
                "prompt_learner": prompt_learner.state_dict(),
                "acceptance_policy": policy.state_dict(),
                "args": vars(args),
            },
            checkpoint_path,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the CARE correction-acceptance policy"
    )
    parser.add_argument("--train_data_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prompt_checkpoint", required=True)
    parser.add_argument("--save_path", default="./checkpoints/care")
    parser.add_argument("--meta_file", default="")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="source batch size; defaults to 2 for VisA and 4 for MVTec AD",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--max_steps_per_epoch", type=int, default=0)
    parser.add_argument("--gnp_rank", type=int, default=64)
    parser.add_argument("--tgrf_temperature", type=float, default=5.0)
    parser.add_argument("--policy_hidden_dim", type=int, default=32)
    parser.add_argument("--acceptance_loss_weight", type=float, default=0.2)
    parser.add_argument("--suppression_weight", type=float, default=0.1)
    args = parser.parse_args()
    if args.batch_size is None:
        source_batch_sizes = {"visa": 2, "mvtec": 4}
        dataset_key = args.dataset.lower()
        if dataset_key not in source_batch_sizes:
            parser.error(
                "--batch_size is required when the source dataset is neither "
                "VisA nor MVTec AD"
            )
        args.batch_size = source_batch_sizes[dataset_key]
    return args


if __name__ == "__main__":
    arguments = parse_args()
    setup_seed(arguments.seed)
    train(arguments)
