"""Evaluate CARE and its text-only and fixed-correction baselines."""

import argparse
import random
import time

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from tabulate import tabulate
from tqdm import tqdm

from dataloader.dataset import Dataset
from model.backbone import compute_similarity, get_similarity_map, load_backbone
from model.care import (
    CorrectionAcceptancePolicy,
    build_candidate_batch,
    build_gnp_batch,
    fuse_with_acceptance,
    risk_features,
)
from model.prompt_ensemble import AnomalyCLIP_PromptLearner
from utils.logger import get_logger
from utils.metrics import image_level_metrics, pixel_level_metrics
from utils.transforms import get_transform


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode_text_features(model, prompt_learner):
    prompts, tokenized_prompts, compound_prompts = prompt_learner(cls_id=None)
    features = model.encode_text_learn(
        prompts, tokenized_prompts, compound_prompts
    ).float()
    features = torch.stack(torch.chunk(features, dim=0, chunks=2), dim=1)
    return features / features.norm(dim=-1, keepdim=True)


def build_text_map(model, image, text_features, args):
    image_features, patch_features = model.encode_image(
        image, args.features_list, DPAM_layer=20
    )
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    image_score = (image_features @ text_features.permute(0, 2, 1) / 0.07)
    image_score = image_score.softmax(dim=-1)[:, 0, 1]

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
            (similarity_map[:, 1] + 1.0 - similarity_map[:, 0]) / 2.0
        )

    text_map = torch.stack(anomaly_maps, dim=0).sum(dim=0)
    return text_map, image_score, selected_patch_features


def prepare_result_store(objects):
    return {
        name: {
            "gt_sp": [],
            "pr_sp": [],
            "imgs_masks": [],
            "anomaly_maps": [],
        }
        for name in objects
    }


def evaluate(args):
    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), design)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device).eval()

    acceptance_policy = None
    if args.method == "care":
        if "acceptance_policy" not in checkpoint:
            raise KeyError(
                "The checkpoint has no 'acceptance_policy'. Use a checkpoint "
                "created by train.py or evaluate with --method text/fixed."
            )
        acceptance_policy = CorrectionAcceptancePolicy(
            input_dim=10, hidden_dim=args.hidden_dim
        ).to(device)
        acceptance_policy.load_state_dict(checkpoint["acceptance_policy"])
        acceptance_policy.eval()

    model.to(device)
    text_features = encode_text_features(model, prompt_learner)
    preprocess, target_transform = get_transform(args)
    dataset = Dataset(
        root=args.data_path,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=args.dataset,
        mode="test",
        meta_file=args.meta_file,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )
    result_store = prepare_result_store(dataset.obj_list)
    acceptance_values = []
    inference_times = []

    for items in tqdm(loader, desc=f"Evaluating {args.method}"):
        image = items["img"].to(device)
        class_name = items["cls_name"][0]
        mask = (items["img_mask"] > 0.5).float()
        result_store[class_name]["imgs_masks"].append(mask)
        result_store[class_name]["gt_sp"].extend(items["anomaly"].cpu())

        with torch.no_grad():
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            text_map, image_score, patch_features = build_text_map(
                model, image, text_features, args
            )
            output_map = text_map

            if args.method in {"fixed", "care"}:
                gnp_map = build_gnp_batch(
                    patch_features,
                    args.image_size,
                    rank=args.gnp_rank,
                    multiscale=args.gnp_multiscale,
                )
                candidate_map = build_candidate_batch(
                    text_map, gnp_map, temperature=args.tgrf_temperature
                )
                output_map = candidate_map
                if args.method == "care":
                    representation = risk_features(
                        text_map, candidate_map, gnp_map
                    )
                    acceptance = acceptance_policy(representation)
                    output_map = fuse_with_acceptance(
                        text_map, candidate_map, acceptance
                    )
                    acceptance_values.extend(
                        acceptance.flatten().detach().cpu().tolist()
                    )

            if device == "cuda":
                torch.cuda.synchronize()
            inference_times.append(time.perf_counter() - start)

        result_store[class_name]["pr_sp"].extend(image_score.detach().cpu())
        smoothed = np.stack(
            [
                gaussian_filter(sample, sigma=args.sigma)
                for sample in output_map.detach().cpu().numpy()
            ]
        )
        result_store[class_name]["anomaly_maps"].append(
            torch.from_numpy(smoothed)
        )

    table = summarize_metrics(result_store, dataset.obj_list, args.metrics, logger)
    logger.info("\n%s", table)
    if acceptance_values:
        values = np.asarray(acceptance_values)
        logger.info(
            "Acceptance weight mean/std/min/max: %.4f / %.4f / %.4f / %.4f",
            values.mean(), values.std(), values.min(), values.max()
        )
    if inference_times:
        milliseconds = np.asarray(inference_times) * 1000.0
        logger.info(
            "Inference time: %.1f +/- %.1f ms/image (%d images)",
            milliseconds.mean(), milliseconds.std(), len(milliseconds)
        )


def summarize_metrics(result_store, objects, metric_scope, logger):
    rows = []
    aggregated = []
    for name in objects:
        values = result_store[name]
        if not values["anomaly_maps"]:
            logger.warning("Skip %s: no evaluated samples found.", name)
            continue
        values["imgs_masks"] = torch.cat(values["imgs_masks"]).numpy()
        values["anomaly_maps"] = torch.cat(values["anomaly_maps"]).numpy()

        if metric_scope == "image-level":
            scores = [
                image_level_metrics(result_store, name, "image-auroc"),
                image_level_metrics(result_store, name, "image-ap"),
            ]
        elif metric_scope == "pixel-level":
            scores = [
                pixel_level_metrics(result_store, name, "pixel-auroc"),
                pixel_level_metrics(result_store, name, "pixel-aupro"),
            ]
        else:
            scores = [
                pixel_level_metrics(result_store, name, "pixel-auroc"),
                pixel_level_metrics(result_store, name, "pixel-aupro"),
                pixel_level_metrics(result_store, name, "pixel-auprc"),
                image_level_metrics(result_store, name, "image-auroc"),
                image_level_metrics(result_store, name, "image-ap"),
            ]
        aggregated.append(scores)
        rows.append([name, *[f"{score * 100:.1f}" for score in scores]])

    headers = {
        "image-level": ["objects", "image_auroc", "image_ap"],
        "pixel-level": ["objects", "pixel_auroc", "pixel_aupro"],
        "image-pixel-level": [
            "objects", "pixel_auroc", "pixel_aupro", "pixel_auprc",
            "image_auroc", "image_ap",
        ],
    }[metric_scope]
    mean_scores = np.nanmean(np.asarray(aggregated, dtype=float), axis=0)
    rows.append(["mean", *[f"{score * 100:.1f}" for score in mean_scores]])
    return tabulate(rows, headers=headers, tablefmt="pipe")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate CARE for zero-shot anomaly detection."
    )
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--save_path", default="results/care")
    parser.add_argument("--meta_file", default=None)
    parser.add_argument(
        "--method", choices=["text", "fixed", "care"], default="care"
    )
    parser.add_argument("--metrics", choices=["image-level", "pixel-level", "image-pixel-level"], default="image-pixel-level")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--gnp_rank", type=int, default=64)
    parser.add_argument("--gnp_multiscale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tgrf_temperature", type=float, default=5.0)
    parser.add_argument("--sigma", type=float, default=13.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=111)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    setup_seed(arguments.seed)
    evaluate(arguments)
