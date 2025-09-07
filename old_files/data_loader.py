import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.manifold import make_swiss_roll
import matplotlib.pyplot as plt


class GaussianClustersDataset(Dataset):
    def __init__(self, n_samples=1000, n_features=2, n_centers=4, cluster_std=1.0, random_state=42):
        self.data, self.labels = make_blobs(
            n_samples=n_samples,
            centers=n_centers,
            n_features=n_features,
            cluster_std=cluster_std,
            random_state=random_state
        )
        self.data = torch.FloatTensor(self.data)
        self.labels = torch.LongTensor(self.labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class LowDimManifoldDataset(Dataset):
    def __init__(self, dataset_type='swiss_roll', n_samples=1000, noise=0.1, random_state=42):
        if dataset_type == 'swiss_roll':
            data, color = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=random_state)
            self.data = torch.FloatTensor(data)
            self.labels = torch.FloatTensor(color)
        elif dataset_type == 'circles':
            data, labels = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
            self.data = torch.FloatTensor(data)
            self.labels = torch.LongTensor(labels)
        elif dataset_type == 'moons':
            data, labels = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
            self.data = torch.FloatTensor(data)
            self.labels = torch.LongTensor(labels)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class HierarchicalDataset(Dataset):
    def __init__(self, n_samples=1000, n_features=10, n_clusters=8, hierarchy_depth=2, random_state=42):
        np.random.seed(random_state)
        
        # Create hierarchical structure
        data = []
        labels = []
        
        for i in range(n_clusters):
            # Main cluster center
            cluster_center = np.random.randn(n_features) * 5
            
            # Create sub-clusters within main cluster
            n_subclusters = np.random.randint(2, 5)
            samples_per_subcluster = n_samples // (n_clusters * n_subclusters)
            
            for j in range(n_subclusters):
                # Sub-cluster center around main center
                subcluster_center = cluster_center + np.random.randn(n_features) * 1.5
                
                # Generate samples for this sub-cluster
                cluster_data = np.random.randn(samples_per_subcluster, n_features) * 0.5 + subcluster_center
                cluster_labels = np.full(samples_per_subcluster, i)
                
                data.append(cluster_data)
                labels.append(cluster_labels)
        
        self.data = torch.FloatTensor(np.vstack(data))
        self.labels = torch.LongTensor(np.concatenate(labels))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_model(model_name, num_classes):
    """Load a simple model for the given dataset"""
    if model_name == 'simple_cnn':
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes, input_channels=3):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(64 * 8 * 8, 128)
                self.fc2 = nn.Linear(128, num_classes)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = x.view(-1, 64 * 8 * 8)
                x = self.dropout(self.relu(self.fc1(x)))
                x = self.fc2(x)
                return x
        
        return SimpleCNN(num_classes)
    
    elif model_name == 'simple_mlp':
        class SimpleMLP(nn.Module):
            def __init__(self, input_dim, num_classes):
                super(SimpleMLP, self).__init__()
                self.fc1 = nn.Linear(input_dim, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, num_classes)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x):
                x = x.view(x.size(0), -1)  # Flatten
                x = self.dropout(self.relu(self.fc1(x)))
                x = self.dropout(self.relu(self.fc2(x)))
                x = self.fc3(x)
                return x
        
        return SimpleMLP(input_dim=784, num_classes=num_classes)  # Default for MNIST-like
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def create_data_loader(args):
    """Create data loader based on arguments"""
    
    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root=args.data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=args.data_dir, train=False, download=True, transform=transform
        )
        num_classes = 10
        input_channels = 1
    
    elif args.dataset == 'fashionmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        
        train_dataset = torchvision.datasets.FashionMNIST(
            root=args.data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=args.data_dir, train=False, download=True, transform=transform
        )
        num_classes = 10
        input_channels = 1
    
    elif args.dataset == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        
        train_dataset = torchvision.datasets.SVHN(
            root=args.data_dir, split='train', download=True, transform=transform
        )
        test_dataset = torchvision.datasets.SVHN(
            root=args.data_dir, split='test', download=True, transform=transform
        )
        num_classes = 10
        input_channels = 3
    
    elif args.dataset == 'gaussian_clusters':
        train_dataset = GaussianClustersDataset(
            n_samples=args.n_samples,
            n_features=args.n_features,
            n_centers=args.n_centers,
            cluster_std=args.cluster_std,
            random_state=42
        )
        # Create test split
        test_dataset = GaussianClustersDataset(
            n_samples=args.n_samples // 4,
            n_features=args.n_features,
            n_centers=args.n_centers,
            cluster_std=args.cluster_std,
            random_state=123
        )
        num_classes = args.n_centers
        input_channels = 1
    
    elif args.dataset == 'low_dim_manifold':
        train_dataset = LowDimManifoldDataset(
            dataset_type=args.manifold_type,
            n_samples=args.n_samples,
            noise=args.noise,
            random_state=42
        )
        test_dataset = LowDimManifoldDataset(
            dataset_type=args.manifold_type,
            n_samples=args.n_samples // 4,
            noise=args.noise,
            random_state=123
        )
        num_classes = 2 if args.manifold_type in ['circles', 'moons'] else 1
        input_channels = 1
    
    elif args.dataset == 'hierarchical':
        train_dataset = HierarchicalDataset(
            n_samples=args.n_samples,
            n_features=args.n_features,
            n_clusters=args.n_centers,
            hierarchy_depth=args.hierarchy_depth,
            random_state=42
        )
        test_dataset = HierarchicalDataset(
            n_samples=args.n_samples // 4,
            n_features=args.n_features,
            n_clusters=args.n_centers,
            hierarchy_depth=args.hierarchy_depth,
            random_state=123
        )
        num_classes = args.n_centers
        input_channels = 1
    
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    return train_loader, test_loader, num_classes, input_channels


def main():
    parser = argparse.ArgumentParser(description='Data Loader with Multiple Datasets')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'fashionmnist', 'svhn', 'gaussian_clusters', 
                               'low_dim_manifold', 'hierarchical'],
                       help='Dataset to use')
    
    parser.add_argument('--data_dir', type=str, default='/Users/tanmoy/research/data',
                       help='Directory to store datasets')
    
    # Data loader arguments
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for data loaders')
    
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loaders')
    
    # Synthetic dataset arguments
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples for synthetic datasets')
    
    parser.add_argument('--n_features', type=int, default=2,
                       help='Number of features for synthetic datasets')
    
    parser.add_argument('--n_centers', type=int, default=4,
                       help='Number of centers/clusters for synthetic datasets')
    
    parser.add_argument('--cluster_std', type=float, default=1.0,
                       help='Standard deviation of clusters')
    
    parser.add_argument('--noise', type=float, default=0.1,
                       help='Noise level for manifold datasets')
    
    parser.add_argument('--manifold_type', type=str, default='swiss_roll',
                       choices=['swiss_roll', 'circles', 'moons'],
                       help='Type of low-dimensional manifold')
    
    parser.add_argument('--hierarchy_depth', type=int, default=2,
                       help='Depth of hierarchical structure')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='simple_mlp',
                       choices=['simple_cnn', 'simple_mlp'],
                       help='Model architecture to use')
    
    # Other arguments
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize the dataset (for 2D synthetic data)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Create data loaders
    train_loader, test_loader, num_classes, input_channels = create_data_loader(args)
    
    # Load model
    if args.dataset in ['mnist', 'fashionmnist', 'svhn']:
        model = load_model(args.model, num_classes)
    else:
        # For synthetic datasets, use MLP
        if args.dataset == 'gaussian_clusters':
            input_dim = args.n_features
        elif args.dataset == 'low_dim_manifold':
            input_dim = 3 if args.manifold_type == 'swiss_roll' else 2
        elif args.dataset == 'hierarchical':
            input_dim = args.n_features
        
        class SimpleMLP(nn.Module):
            def __init__(self, input_dim, num_classes):
                super(SimpleMLP, self).__init__()
                self.fc1 = nn.Linear(input_dim, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, num_classes)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x):
                x = x.view(x.size(0), -1)
                x = self.dropout(self.relu(self.fc1(x)))
                x = self.dropout(self.relu(self.fc2(x)))
                x = self.fc3(x)
                return x
        
        model = SimpleMLP(input_dim, num_classes)
    
    print(f"Dataset: {args.dataset}")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Model: {model}")
    
    # Visualize synthetic datasets if requested
    if args.visualize and args.dataset in ['gaussian_clusters', 'low_dim_manifold', 'hierarchical']:
        plt.figure(figsize=(10, 8))
        
        # Get a batch of data
        data_iter = iter(train_loader)
        data, labels = next(data_iter)
        
        if data.shape[1] >= 2:  # At least 2D data
            plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10', alpha=0.7)
            plt.colorbar()
            plt.title(f'{args.dataset.replace("_", " ").title()} Dataset')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.show()
    
    return train_loader, test_loader, model, num_classes


if __name__ == "__main__":
    train_loader, test_loader, model, num_classes = main()