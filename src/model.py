import torch
import torch.nn as nn
from torchvision.models import resnet18

def create_model(num_classes=9):
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
