import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from dataset import MembershipDataset, PrivOutDataset, TaskDataset
from metrics import calculate_tpr_at_fpr
from model import create_model
from rmia import get_rmia_scores
from train import train_reference_models
from utils import create_reference_datasets


MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]


def expose_pickle_classes():
    main_module = sys.modules["__main__"]
    main_module.TaskDataset = TaskDataset
    main_module.MembershipDataset = MembershipDataset


def build_transform():
    return transforms.Compose([
        transforms.Resize(32),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def subset_indices(indices, labels, size, seed):
    if size is None or size >= len(indices):
        return indices
    _, chosen = train_test_split(
        np.asarray(indices),
        test_size=size,
        stratify=np.asarray(labels)[indices],
        random_state=seed,
    )
    return np.asarray(chosen)


def validate_submission(output_csv, priv_data):
    df = pd.read_csv(output_csv)
    expected_ids = {int(id_) for id_ in priv_data.ids}
    actual_ids = set(df["id"].astype(int))
    if list(df.columns) != ["id", "score"]:
        raise ValueError(f"submission columns must be ['id', 'score'], got {list(df.columns)}")
    if len(df) != len(priv_data):
        raise ValueError(f"submission has {len(df)} rows, expected {len(priv_data)}")
    if df["id"].duplicated().any():
        raise ValueError("submission contains duplicate ids")
    if actual_ids != expected_ids:
        raise ValueError("submission ids do not match priv.pt ids")
    if not pd.api.types.is_numeric_dtype(df["score"]):
        raise ValueError("submission scores must be numeric")
    if not df["score"].between(0.0, 1.0).all():
        raise ValueError("submission scores must be in [0, 1]")


def main():
    parser = argparse.ArgumentParser(description="Run adapted RMIA for Assignment 1.")
    parser.add_argument("--directory", type=Path, default=Path("."), help="Directory containing pub.pt, priv.pt, and model.pt")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"), help="Output CSV path")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for each reference model")
    parser.add_argument("--num_ref_models", type=int, default=4, help="Number of reference models")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training and scoring")
    parser.add_argument("--gamma", type=float, default=1.0, help="RMIA likelihood ratio threshold")
    parser.add_argument("--a", type=float, default=0.3, help="RMIA smoothing factor")
    parser.add_argument("--Z_size", type=int, default=1000, help="Number of public samples used for RMIA Z")
    parser.add_argument("--val_size", type=float, default=0.2, help="Fraction of pub.pt used for validation")
    parser.add_argument("--val_limit", type=int, default=None, help="Optional stratified validation limit for smoke tests")
    parser.add_argument("--priv_limit", type=int, default=None, help="Optional private limit for smoke tests")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    expose_pickle_classes()

    directory = args.directory.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = build_transform()

    print("Loading datasets...")
    pub_data = torch.load(directory / "pub.pt", map_location="cpu", weights_only=False)
    priv_loaded = torch.load(directory / "priv.pt", map_location="cpu", weights_only=False)
    pub_data.transform = transform

    priv_data = PrivOutDataset(
        ids=priv_loaded.ids,
        imgs=priv_loaded.imgs,
        labels=priv_loaded.labels,
        transform=transform,
    )

    pub_membership = np.asarray(pub_data.membership, dtype=int)
    print(f"pub.pt: {len(pub_data)} samples, members={int(pub_membership.sum())}, non_members={int((pub_membership == 0).sum())}")
    print(f"priv.pt: {len(priv_data)} samples")

    print("Loading target model...")
    model = create_model(num_classes=9).to(device)
    state_dict = torch.load(directory / "model.pt", map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"model load mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()

    indices = np.arange(len(pub_data))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=args.val_size,
        stratify=pub_membership,
        random_state=args.seed,
    )
    val_indices = subset_indices(val_indices, pub_membership, args.val_limit, args.seed)

    z_pool = np.asarray(list(set(indices) - set(val_indices)))
    z_size = min(args.Z_size, len(z_pool))
    z_indices = np.random.default_rng(args.seed).choice(z_pool, size=z_size, replace=False)

    val_loader = DataLoader(Subset(pub_data, val_indices), batch_size=args.batch_size, shuffle=False)
    z_loader = DataLoader(Subset(pub_data, z_indices), batch_size=args.batch_size, shuffle=False)

    reference_datasets, _ = create_reference_datasets(
        pub_data,
        num_ref_models=args.num_ref_models,
        random_state=args.seed,
    )
    reference_models = train_reference_models(
        reference_datasets,
        device,
        epochs=args.epochs,
        num_classes=9,
        batch_size=args.batch_size,
    )

    print("Computing validation RMIA scores...")
    val_scores = get_rmia_scores(
        model,
        reference_models,
        val_loader,
        z_loader,
        gamma=args.gamma,
        a=args.a,
        device=device,
    )
    id_to_membership = {int(id_): int(membership) for id_, membership in zip(pub_data.ids, pub_data.membership)}
    y_true = [id_to_membership[id_] for id_ in val_scores.keys()]
    y_score = [val_scores[id_] for id_ in val_scores.keys()]
    print(f"Validation AUC: {roc_auc_score(y_true, y_score):.6f}")
    print(f"TPR@5%FPR: {calculate_tpr_at_fpr(y_true, y_score, target_fpr=0.05):.6f}")

    priv_indices = np.arange(len(priv_data))
    if args.priv_limit is not None:
        priv_indices = priv_indices[:args.priv_limit]
    priv_loader = DataLoader(Subset(priv_data, priv_indices), batch_size=args.batch_size, shuffle=False)

    print("Computing private RMIA scores...")
    priv_scores = get_rmia_scores(
        model,
        reference_models,
        priv_loader,
        z_loader,
        gamma=args.gamma,
        a=args.a,
        device=device,
    )

    score_values = np.asarray([priv_scores[int(priv_data.ids[i])] for i in priv_indices], dtype=float)
    score_values = np.clip(score_values, 0.0, 1.0)

    output_csv = args.output if args.output.is_absolute() else directory / args.output
    df = pd.DataFrame({
        "id": [int(priv_data.ids[i]) for i in priv_indices],
        "score": score_values,
    })
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

    if args.priv_limit is None:
        validate_submission(output_csv, priv_data)
        print("Submission validation passed.")
    else:
        print("Skipped full submission validation because --priv_limit was set.")


if __name__ == "__main__":
    main()
