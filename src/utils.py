import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Function to create reference datasets
def create_reference_datasets(pub_data, num_ref_models=5, test_size=0.5, random_state=42):
    reference_datasets = []
    ref_inclusion_lists = []
    indices = np.arange(len(pub_data))
    rng = np.random.default_rng(random_state)
    stratify = getattr(pub_data, "labels", None)
    for _ in range(num_ref_models):
        train_indices, _ = train_test_split(
            indices,
            test_size=test_size,
            random_state=int(rng.integers(0, 1_000_000)),
            stratify=stratify
        )
        ref_dataset = Subset(pub_data, train_indices)
        reference_datasets.append(ref_dataset)
        ref_inclusion_lists.append(train_indices)
    return reference_datasets, ref_inclusion_lists

# Function to compute probabilities Pr(x|θ) for a model and data_loader
def compute_model_probabilities(model, data_loader, device=None):
    if device is None:
        device = next(model.parameters()).device
    probs_dict = {}
    model.eval()
    with torch.no_grad():
        for ids, imgs, labels, *_ in tqdm(data_loader, desc="Scoring", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device, dtype=torch.long)
            outputs = F.softmax(model(imgs), dim=1)
            probs = outputs[torch.arange(outputs.shape[0], device=device), labels]
            for id_, prob in zip(ids, probs):
                if hasattr(id_, "item"):
                    id_ = id_.item()
                probs_dict[int(id_)] = prob.item()
    return probs_dict
