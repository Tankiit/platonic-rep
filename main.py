import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision import datasets, transforms

from tqdm import tqdm
from torch.utils.data import DataLoader

import argparse
from agop_computation import (
    compute_agop_efficient,
    compute_agop_streaming,
    compute_agop_with_hooks,
    create_layer_extractor_for_sequential,
    create_forward_from_layer_for_sequential
)


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "maskrcnn"])
parser.add_argument("--dataset", type=str, default="cifar10", choices=["imagenet", "cifar10", "cifar100"])
parser.add_argument("--layer", type=int, default=0)
parser.add_argument("--lambda_reg", type=float, default=1e-4)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_samples", type=int, default=1024)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--data_dir", type=str, default="/Users/cril/tanmoy/research/data")
parser.add_argument("--output_dir", type=str, default="./results")
args = parser.parse_args()

if args.device is None:
    if torch.backends.mps.is_available():
        args.device = "mps"
    elif torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"

if args.model == "resnet50":
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
elif args.model == "maskrcnn":
    model = MaskRCNN(backbone=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2))

return_nodes = {
    "layer1": "layer1",
    "layer2": "layer2",
    "layer3": "layer3",
    "layer4": "layer4",
}

feature_extractor = create_feature_extractor(
    model,
    return_nodes=return_nodes
)


if args.dataset == "cifar10":
    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = model.to(args.device)
    model.eval()
    
    for data, labels in tqdm(train_loader):
        data = data.to(args.device)
        labels = labels.to(args.device)
        features = feature_extractor(data)
        features = features["layer1"]
        features = features.view(features.size(0), -1)
        print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

elif args.dataset == "imagenet":
    train_dataset = datasets.ImageNet(root=args.data_dir, split="train", download=True, transform=transforms.ToTensor())
    test_dataset = datasets.ImageNet(root=args.data_dir, split="val", download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


