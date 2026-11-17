"""Train the source-domain object-agnostic prompts used by CARE."""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataloader.dataset import Dataset
from model.backbone import compute_similarity, get_similarity_map, load_backbone
from model.prompt_ensemble import AnomalyCLIP_PromptLearner
from utils.logger import get_logger
from utils.losses import BinaryDiceLoss, FocalLoss
from utils.transforms import get_transform


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess, target_transform = get_transform(args)
    design = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
    }

    model, _ = load_backbone(
        "ViT-L/14@336px", device=device, design_details=design
    )
    model.eval()
    model.visual.DAPM_replace(DPAM_layer=20)
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
    )

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), design).to(device)
    model.to(device)
    optimizer = torch.optim.Adam(
        prompt_learner.parameters(),
        lr=args.learning_rate,
        betas=(0.5, 0.999),
    )
    focal_loss = FocalLoss()
    dice_loss = BinaryDiceLoss()

    for epoch in range(args.epochs):
        prompt_learner.train()
        localization_losses = []
        classification_losses = []
        for batch in tqdm(loader, desc=f"prompt epoch {epoch + 1}"):
            image = batch["img"].to(device)
            label = batch["anomaly"].long().to(device)
            target = (batch["img_mask"].to(device) > 0.5).float()

            with torch.no_grad():
                image_features, patch_features = model.encode_image(
                    image, args.features_list, DPAM_layer=20
                )
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

            prompts, tokens, compound_prompts = prompt_learner(cls_id=None)
            text_features = model.encode_text_learn(
                prompts, tokens, compound_prompts
            ).float()
            text_features = torch.stack(
                torch.chunk(text_features, dim=0, chunks=2), dim=1
            )
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )
            image_logits = (
                image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            )[:, 0] / 0.07
            classification_loss = F.cross_entropy(image_logits, label)

            similarity_maps = []
            for index, patch_feature in enumerate(patch_features):
                if index < args.feature_map_layer[0]:
                    continue
                patch_feature = patch_feature / patch_feature.norm(
                    dim=-1, keepdim=True
                )
                similarity, _ = compute_similarity(
                    patch_feature, text_features[0]
                )
                similarity_maps.append(
                    get_similarity_map(
                        similarity[:, 1:, :], args.image_size
                    ).permute(0, 3, 1, 2)
                )

            localization_loss = torch.zeros((), device=device)
            for similarity_map in similarity_maps:
                localization_loss = localization_loss + focal_loss(
                    similarity_map, target
                )
                localization_loss = localization_loss + dice_loss(
                    similarity_map[:, 1], target
                )
                localization_loss = localization_loss + dice_loss(
                    similarity_map[:, 0], 1.0 - target
                )
            localization_loss = args.localization_weight * localization_loss
            total_loss = localization_loss + classification_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            localization_losses.append(float(localization_loss.detach().cpu()))
            classification_losses.append(float(classification_loss.detach().cpu()))

        logger.info(
            "epoch [%d/%d], localization_loss=%.4f, image_loss=%.4f",
            epoch + 1,
            args.epochs,
            float(np.mean(localization_losses)),
            float(np.mean(classification_losses)),
        )
        torch.save(
            {"prompt_learner": prompt_learner.state_dict()},
            os.path.join(args.save_path, f"epoch_{epoch + 1}.pth"),
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train source-domain prompts for CARE"
    )
    parser.add_argument("--train_data_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--save_path", default="checkpoints/prompts")
    parser.add_argument("--meta_file", default="")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--localization_weight", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=111)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    setup_seed(arguments.seed)
    train(arguments)
