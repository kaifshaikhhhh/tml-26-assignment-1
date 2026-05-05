import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import create_model

def train_reference_models(reference_datasets, device, epochs=5, num_classes=9, batch_size=64, lr=0.001):
    reference_models = []
    for e, dataset in enumerate(reference_datasets):
        print(f"Training reference model {e+1}/{len(reference_datasets)}")
        model_ref = create_model(num_classes=num_classes)
        model_ref = model_ref.to(device)
        optimizer = torch.optim.Adam(model_ref.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in tqdm(range(epochs)):
            model_ref.train()
            for ids, imgs, labels, memberships in loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model_ref(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        reference_models.append(model_ref)
    return reference_models
