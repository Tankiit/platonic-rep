#!/usr/bin/env python3
"""
Multi-Model Analysis: Macroscopic and Mesoscopic Analysis across different architectures
Supports MLP, CNN, ResNet, ViT models using timm library with CIFAR-10/100, SVHN datasets
"""

import torch
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple, Optional
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# TensorBoard imports
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")

# Import timm for model loading
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models.feature_extraction import create_feature_extractor

# Import our analysis modules
from macroscopic import MacroscopicAnalysis
from mesoscopic import MesoscopicAnalysis

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class MultiModelAnalyzer:
    """
    Comprehensive analyzer for multiple model architectures and datasets
    """
    
    def __init__(self, output_dir="./results/multi_model_analysis/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect device
        self.device = self._auto_detect_device()
        print(f"Auto-detected device: {self.device}")
        
        # Initialize TensorBoard logging
        self.tensorboard_dir = self.output_dir / "tensorboard_logs"
        self.tensorboard_dir.mkdir(exist_ok=True)
        
        if TENSORBOARD_AVAILABLE:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir / f"run_{timestamp}"))
            print(f"TensorBoard logging enabled: {self.tensorboard_dir}")
        else:
            self.writer = None
            print("TensorBoard logging disabled")
        
        # Supported model architectures
        self.supported_models = {
            'mlp': ['mlp_mixer_b16_224', 'mlp_mixer_b32_224'],
            'cnn': ['convnext_tiny', 'convnext_small', 'efficientnet_b0'],
            'resnet': ['resnet18', 'resnet34', 'resnet50'],
            'vit': ['vit_base_patch16_224', 'vit_small_patch16_224', 'deit_base_patch16_224']
        }
        
        # Supported datasets
        self.supported_datasets = ['cifar10', 'cifar100', 'svhn']
        
        # Dataset configurations
        self.dataset_configs = {
            'cifar10': {'num_classes': 10, 'input_size': 32, 'channels': 3},
            'cifar100': {'num_classes': 100, 'input_size': 32, 'channels': 3},
            'svhn': {'num_classes': 10, 'input_size': 32, 'channels': 3}
        }
        
        # Experiment tracking
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_log = {
            'experiment_id': self.experiment_id,
            'start_time': datetime.now().isoformat(),
            'device': str(self.device),
            'models_analyzed': [],
            'datasets_analyzed': [],
            'total_samples_processed': 0,
            'analysis_duration': 0
        }
        
    def get_available_models(self, model_type: str = None) -> List[str]:
        """Get list of available models from timm"""
        if model_type:
            return self.supported_models.get(model_type, [])
        
        all_models = []
        for models in self.supported_models.values():
            all_models.extend(models)
        return all_models
    
    def _auto_detect_device(self) -> torch.device:
        """Auto-detect the best available device (CUDA > MPS > CPU)"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("MPS (Apple Silicon) available")
        else:
            device = torch.device('cpu')
            print("Using CPU")
        
        return device
    
    def load_model(self, model_name: str, dataset: str, pretrained: bool = True) -> torch.nn.Module:
        """Load model with appropriate configuration for dataset"""
        try:
            # Get dataset config
            dataset_config = self.dataset_configs[dataset]
            num_classes = dataset_config['num_classes']
            
            # Create model
            if pretrained:
                model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
            else:
                model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
            
            # Adjust input size for CIFAR/SVHN if needed
            if dataset in ['cifar10', 'cifar100', 'svhn']:
                # Some models expect 224x224, but CIFAR/SVHN are 32x32
                # We'll use interpolation to handle this
                pass
            
            return model
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None
    
    def get_feature_extractor(self, model: torch.nn.Module, model_name: str) -> torch.nn.Module:
        """Create feature extractor for different model architectures"""
        try:
            if 'resnet' in model_name.lower():
                # ResNet: extract features after each stage
                return_nodes = {
                    'layer1': 'stage1',
                    'layer2': 'stage2', 
                    'layer3': 'stage3',
                    'layer4': 'stage4'
                }
            elif 'vit' in model_name.lower() or 'deit' in model_name.lower():
                # ViT: extract features after each transformer block
                num_blocks = len(model.blocks)
                return_nodes = {f'blocks.{i}': f'block_{i}' for i in range(num_blocks)}
            elif 'convnext' in model_name.lower():
                # ConvNeXt: extract features after each stage
                return_nodes = {
                    'stages.0': 'stage1',
                    'stages.1': 'stage2',
                    'stages.2': 'stage3',
                    'stages.3': 'stage4'
                }
            elif 'mlp_mixer' in model_name.lower():
                # MLP-Mixer: extract features after each mixer block
                num_blocks = len(model.blocks)
                return_nodes = {f'blocks.{i}': f'block_{i}' for i in range(num_blocks)}
            else:
                # Default: try to extract from common layer names
                return_nodes = {
                    'features': 'features',
                    'classifier': 'classifier'
                }
            
            return create_feature_extractor(model, return_nodes)
            
        except Exception as e:
            print(f"Error creating feature extractor for {model_name}: {e}")
            return model
    
    def create_data_loader(self, dataset: str, batch_size: int = 32, split: str = 'train') -> Tuple[torch.utils.data.DataLoader, int]:
        """Create data loader for specified dataset"""
        try:
            from torchvision import datasets, transforms
            
            # Dataset-specific transforms
            if dataset == 'cifar10':
                transform = transforms.Compose([
                    transforms.Resize(224),  # Resize to match model input
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
                dataset_obj = datasets.CIFAR10('/Users/tanmoy/research/data', train=(split=='train'), download=False, transform=transform)
                num_classes = 10
                
            elif dataset == 'cifar100':
                transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                ])
                dataset_obj = datasets.CIFAR100('/Users/tanmoy/research/data', train=(split=='train'), download=False, transform=transform)
                num_classes = 100
                
            elif dataset == 'svhn':
                transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
                ])
                dataset_obj = datasets.SVHN('/Users/tanmoy/research/data', split=split, download=False, transform=transform)
                num_classes = 10
                
            else:
                raise ValueError(f"Unsupported dataset: {dataset}")
            
            # Create data loader
            data_loader = torch.utils.data.DataLoader(
                dataset_obj, batch_size=batch_size, shuffle=(split=='train'), 
                num_workers=2, pin_memory=True
            )
            
            return data_loader, num_classes
            
        except Exception as e:
            print(f"Error creating data loader for {dataset}: {e}")
            return None, 0
    
    def extract_features(self, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
                        device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Extract features from all layers of the model"""
        model.eval()
        model = model.to(device)
        
        all_features = {}
        batch_features = []
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(tqdm(data_loader, desc="Extracting features")):
                data = data.to(device)
                
                # Forward pass
                features = model(data)
                
                # Process features based on model type
                if isinstance(features, dict):
                    # Feature extractor returns dict
                    for layer_name, layer_features in features.items():
                        if layer_name not in all_features:
                            all_features[layer_name] = []
                        
                        # Global average pooling for spatial features
                        if len(layer_features.shape) == 4:  # [B, C, H, W]
                            layer_features = torch.nn.functional.adaptive_avg_pool2d(layer_features, 1).squeeze(-1).squeeze(-1)
                        
                        all_features[layer_name].append(layer_features.cpu())
                        
                else:
                    # Single output
                    if 'features' not in all_features:
                        all_features['features'] = []
                    all_features['features'].append(features.cpu())
                
                batch_features.append(targets.cpu())
                
                # Limit number of samples for analysis
                if batch_idx >= 50:  # ~1600 samples with batch_size=32
                    break
        
        # Concatenate features across batches
        final_features = {}
        for layer_name, layer_features in all_features.items():
            final_features[layer_name] = torch.cat(layer_features, dim=0)
        
        # Stack features across layers: [N, L, D]
        layer_names = sorted(final_features.keys())
        feature_dim = final_features[layer_names[0]].shape[1]
        num_samples = final_features[layer_names[0]].shape[0]
        
        stacked_features = torch.zeros(num_samples, len(layer_names), feature_dim)
        for i, layer_name in enumerate(layer_names):
            stacked_features[:, i, :] = final_features[layer_name]
        
        return {
            'feats': stacked_features,
            'layer_names': layer_names,
            'targets': torch.cat(batch_features, dim=0)
        }
    
    def run_analysis(self, model_name: str, dataset: str, pretrained: bool = True, 
                    save_features: bool = True, device: str = None) -> Dict:
        """Run complete analysis for a model-dataset combination"""
        start_time = time.time()
        
        # Use auto-detected device if none specified
        if device is None:
            device = self.device
        
        print(f"\n=== Analyzing {model_name} on {dataset} ===")
        print(f"Using device: {device}")
        
        # Log experiment start
        if self.writer:
            self.writer.add_text(f"experiment/{model_name}_{dataset}/start", 
                               f"Started analysis at {datetime.now().isoformat()}")
        
        # Load model
        print(f"Loading model {model_name}...")
        model = self.load_model(model_name, dataset, pretrained)
        if model is None:
            return None
        
        # Create feature extractor
        print(f"Creating feature extractor...")
        feature_extractor = self.get_feature_extractor(model, model_name)
        
        # Create data loader
        print(f"Creating data loader for {dataset}...")
        data_loader, num_classes = self.create_data_loader(dataset, batch_size=32, split='train')
        if data_loader is None:
            return None
        
        # Extract features
        print(f"Extracting features...")
        features_data = self.extract_features(feature_extractor, data_loader, device)
        
        # Save features if requested
        if save_features:
            features_path = self.output_dir / f"{model_name}_{dataset}_features.pt"
            torch.save(features_data, features_path)
            print(f"Features saved to {features_path}")
        
        # Run macroscopic analysis
        print(f"Running macroscopic analysis...")
        macroscopic_analyzer = MacroscopicAnalysis()
        macroscopic_results = macroscopic_analyzer.analyze_model_from_features(
            features_data, model_name, dataset
        )
        
        # Run mesoscopic analysis
        print(f"Running mesoscopic analysis...")
        mesoscopic_analyzer = MesoscopicAnalysis()
        mesoscopic_results = mesoscopic_analyzer.analyze_model_from_features(
            features_data, model_name, dataset
        )
        
        # Combine results
        combined_results = {
            'model': model_name,
            'dataset': dataset,
            'pretrained': pretrained,
            'macroscopic': macroscopic_results,
            'mesoscopic': mesoscopic_results,
            'metadata': {
                'num_samples': features_data['feats'].shape[0],
                'num_layers': features_data['feats'].shape[1],
                'feature_dim': features_data['feats'].shape[2],
                'layer_names': features_data['layer_names']
            }
        }
        
        # Save combined results
        results_path = self.output_dir / f"{model_name}_{dataset}_analysis.json"
        with open(results_path, 'w') as f:
            json.dump(combined_results, f, indent=2, cls=NumpyEncoder)
        
        # Log to TensorBoard
        if self.writer:
            self._log_analysis_results(combined_results, model_name, dataset)
        
        # Update experiment log
        analysis_duration = time.time() - start_time
        self.experiment_log['models_analyzed'].append(model_name)
        self.experiment_log['datasets_analyzed'].append(dataset)
        self.experiment_log['total_samples_processed'] += features_data['feats'].shape[0]
        
        print(f"Analysis complete. Results saved to {results_path}")
        print(f"Analysis duration: {analysis_duration:.2f} seconds")
        
        return combined_results
    
    def _log_analysis_results(self, results: Dict, model_name: str, dataset: str):
        """Log analysis results to TensorBoard"""
        if not self.writer:
            return
        
        # Log metadata
        metadata = results.get('metadata', {})
        self.writer.add_scalar(f'{model_name}_{dataset}/metadata/num_samples', 
                              metadata.get('num_samples', 0))
        self.writer.add_scalar(f'{model_name}_{dataset}/metadata/num_layers', 
                              metadata.get('num_layers', 0))
        self.writer.add_scalar(f'{model_name}_{dataset}/metadata/feature_dim', 
                              metadata.get('feature_dim', 0))
        
        # Log macroscopic metrics
        if 'macroscopic' in results:
            macro = results['macroscopic']
            if 'information_flow' in macro and 'summary' in macro['information_flow']:
                summary = macro['information_flow']['summary']
                self.writer.add_scalar(f'{model_name}_{dataset}/macroscopic/total_compression', 
                                      summary.get('total_compression', 0))
                self.writer.add_scalar(f'{model_name}_{dataset}/macroscopic/total_task_info_gain', 
                                      summary.get('total_task_info_gain', 0))
                self.writer.add_scalar(f'{model_name}_{dataset}/macroscopic/peak_task_info', 
                                      summary.get('peak_task_info', 0))
                self.writer.add_scalar(f'{model_name}_{dataset}/macroscopic/peak_task_info_layer', 
                                      summary.get('peak_task_info_layer', 0))
        
        # Log mesoscopic metrics
        if 'mesoscopic' in results:
            meso = results['mesoscopic']
            if 'ntk' in meso and 'evolution' in meso['ntk']:
                ntk_evo = meso['ntk']['evolution']
                self.writer.add_scalar(f'{model_name}_{dataset}/mesoscopic/rank_compression', 
                                      ntk_evo.get('rank_compression', 0))
                self.writer.add_scalar(f'{model_name}_{dataset}/mesoscopic/eigenvalue_concentration', 
                                      ntk_evo.get('eigenvalue_concentration', 0))
                self.writer.add_scalar(f'{model_name}_{dataset}/mesoscopic/spectral_sharpening', 
                                      ntk_evo.get('spectral_sharpening', 1))
            
            if 'dynamics' in meso:
                dynamics = meso['dynamics']
                avg_intrinsic = np.mean(dynamics.get('intrinsic_dimension', [0]))
                avg_complexity = np.mean(dynamics.get('feature_complexity', [0]))
                self.writer.add_scalar(f'{model_name}_{dataset}/mesoscopic/avg_intrinsic_dimension', 
                                      avg_intrinsic)
                self.writer.add_scalar(f'{model_name}_{dataset}/mesoscopic/avg_feature_complexity', 
                                      avg_complexity)
        
        # Log layer-wise information if available
        if 'macroscopic' in results and 'information_flow' in results['macroscopic']:
            layers = results['macroscopic']['information_flow'].get('layers', {})
            for layer_name, layer_data in layers.items():
                self.writer.add_scalar(f'{model_name}_{dataset}/layers/{layer_name}/I_X_T', 
                                      layer_data.get('I_X_T', 0))
                self.writer.add_scalar(f'{model_name}_{dataset}/layers/{layer_name}/I_Y_T', 
                                      layer_data.get('I_Y_T', 0))
                self.writer.add_scalar(f'{model_name}_{dataset}/layers/{layer_name}/efficiency', 
                                      layer_data.get('efficiency', 0))
        
        self.writer.flush()
    
    def run_comprehensive_analysis(self, models: List[str], datasets: List[str], 
                                 pretrained: bool = True, device: str = None) -> Dict:
        """Run analysis across multiple models and datasets"""
        all_results = {}
        
        for dataset in datasets:
            all_results[dataset] = {}
            for model_name in models:
                try:
                    result = self.run_analysis(model_name, dataset, pretrained, True, device)
                    if result:
                        all_results[dataset][model_name] = result
                except Exception as e:
                    print(f"Error analyzing {model_name} on {dataset}: {e}")
                    continue
        
        # Generate comparative analysis
        self.generate_comparative_analysis(all_results)
        
        # Finalize experiment logging
        self._finalize_experiment_log()
        
        return all_results
    
    def generate_comparative_analysis(self, all_results: Dict):
        """Generate comparative visualizations across models and datasets"""
        print("\n=== Generating Comparative Analysis ===")
        
        # Create comparison plots
        self.plot_model_comparison(all_results)
        self.plot_dataset_comparison(all_results)
        self.plot_architecture_comparison(all_results)
    
    def plot_model_comparison(self, all_results: Dict):
        """Plot comparison across models for each dataset"""
        for dataset in all_results:
            if not all_results[dataset]:
                continue
                
            models = list(all_results[dataset].keys())
            if len(models) < 2:
                continue
            
            # Macroscopic comparison
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Information flow comparison
            ax = axes[0, 0]
            compressions = []
            task_gains = []
            model_labels = []
            
            for model in models:
                if 'macroscopic' in all_results[dataset][model]:
                    summary = all_results[dataset][model]['macroscopic'].get('information_flow', {}).get('summary', {})
                    compressions.append(summary.get('total_compression', 0))
                    task_gains.append(summary.get('total_task_info_gain', 0))
                    model_labels.append(model[:15])
            
            if compressions:
                ax.scatter(compressions, task_gains, s=100, alpha=0.7)
                for i, label in enumerate(model_labels):
                    ax.annotate(label, (compressions[i], task_gains[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                ax.set_xlabel('Total Compression')
                ax.set_ylabel('Task Info Gain')
                ax.set_title(f'Information Flow - {dataset.upper()}')
                ax.grid(True, alpha=0.3)
            
            # NTK properties comparison
            ax = axes[0, 1]
            rank_compressions = []
            spectral_sharpening = []
            
            for model in models:
                if 'mesoscopic' in all_results[dataset][model]:
                    ntk_evo = all_results[dataset][model]['mesoscopic'].get('ntk', {}).get('evolution', {})
                    rank_compressions.append(ntk_evo.get('rank_compression', 0))
                    spectral_sharpening.append(ntk_evo.get('spectral_sharpening', 1))
            
            if rank_compressions:
                ax.scatter(rank_compressions, spectral_sharpening, s=100, alpha=0.7)
                for i, label in enumerate(model_labels):
                    ax.annotate(label, (rank_compressions[i], spectral_sharpening[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                ax.set_xlabel('NTK Rank Compression')
                ax.set_ylabel('Spectral Sharpening')
                ax.set_title(f'NTK Properties - {dataset.upper()}')
                ax.grid(True, alpha=0.3)
            
            # Feature dynamics comparison
            ax = axes[1, 0]
            intrinsic_dims = []
            complexities = []
            
            for model in models:
                if 'mesoscopic' in all_results[dataset][model]:
                    dynamics = all_results[dataset][model]['mesoscopic'].get('dynamics', {})
                    avg_intrinsic = np.mean(dynamics.get('intrinsic_dimension', [0]))
                    avg_complexity = np.mean(dynamics.get('feature_complexity', [0]))
                    intrinsic_dims.append(avg_intrinsic)
                    complexities.append(avg_complexity)
            
            if intrinsic_dims:
                ax.scatter(intrinsic_dims, complexities, s=100, alpha=0.7)
                for i, label in enumerate(model_labels):
                    ax.annotate(label, (intrinsic_dims[i], complexities[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                ax.set_xlabel('Avg Intrinsic Dimension')
                ax.set_ylabel('Avg Feature Complexity')
                ax.set_title(f'Feature Dynamics - {dataset.upper()}')
                ax.grid(True, alpha=0.3)
            
            # Summary statistics
            ax = axes[1, 1]
            ax.axis('off')
            
            summary_text = f"Dataset: {dataset.upper()}\nModels: {len(models)}\n\n"
            for model in models:
                summary_text += f"{model[:20]}...\n"
                if 'macroscopic' in all_results[dataset][model]:
                    summary = all_results[dataset][model]['macroscopic'].get('information_flow', {}).get('summary', {})
                    summary_text += f"  Compression: {summary.get('total_compression', 0):.3f}\n"
                    summary_text += f"  Task Info: {summary.get('total_task_info_gain', 0):.3f}\n"
                summary_text += "\n"
            
            ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace')
            
            plt.suptitle(f'Model Comparison - {dataset.upper()}', fontsize=16)
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{dataset}_model_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    def plot_dataset_comparison(self, all_results: Dict):
        """Plot comparison across datasets for each model"""
        # Group results by model
        model_results = {}
        for dataset in all_results:
            for model in all_results[dataset]:
                if model not in model_results:
                    model_results[model] = {}
                model_results[model][dataset] = all_results[dataset][model]
        
        # Plot for each model
        for model in model_results:
            if len(model_results[model]) < 2:
                continue
                
            datasets = list(model_results[model].keys())
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Information flow across datasets
            ax = axes[0, 0]
            compressions = []
            task_gains = []
            
            for dataset in datasets:
                if 'macroscopic' in model_results[model][dataset]:
                    summary = model_results[model][dataset]['macroscopic'].get('information_flow', {}).get('summary', {})
                    compressions.append(summary.get('total_compression', 0))
                    task_gains.append(summary.get('total_task_info_gain', 0))
            
            if compressions:
                ax.scatter(compressions, task_gains, s=100, alpha=0.7)
                for i, dataset in enumerate(datasets):
                    ax.annotate(dataset.upper(), (compressions[i], task_gains[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                ax.set_xlabel('Total Compression')
                ax.set_ylabel('Task Info Gain')
                ax.set_title(f'Information Flow - {model}')
                ax.grid(True, alpha=0.3)
            
            # NTK properties across datasets
            ax = axes[0, 1]
            rank_compressions = []
            spectral_sharpening = []
            
            for dataset in datasets:
                if 'mesoscopic' in model_results[model][dataset]:
                    ntk_evo = model_results[model][dataset]['mesoscopic'].get('ntk', {}).get('evolution', {})
                    rank_compressions.append(ntk_evo.get('rank_compression', 0))
                    spectral_sharpening.append(ntk_evo.get('spectral_sharpening', 1))
            
            if rank_compressions:
                ax.scatter(rank_compressions, spectral_sharpening, s=100, alpha=0.7)
                for i, dataset in enumerate(datasets):
                    ax.annotate(dataset.upper(), (rank_compressions[i], spectral_sharpening[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                ax.set_xlabel('NTK Rank Compression')
                ax.set_ylabel('Spectral Sharpening')
                ax.set_title(f'NTK Properties - {model}')
                ax.grid(True, alpha=0.3)
            
            # Feature dynamics across datasets
            ax = axes[1, 0]
            intrinsic_dims = []
            complexities = []
            
            for dataset in datasets:
                if 'mesoscopic' in model_results[model][dataset]:
                    dynamics = model_results[model][dataset]['mesoscopic'].get('dynamics', {})
                    avg_intrinsic = np.mean(dynamics.get('intrinsic_dimension', [0]))
                    avg_complexity = np.mean(dynamics.get('feature_complexity', [0]))
                    intrinsic_dims.append(avg_intrinsic)
                    complexities.append(avg_complexity)
            
            if intrinsic_dims:
                ax.scatter(intrinsic_dims, complexities, s=100, alpha=0.7)
                for i, dataset in enumerate(datasets):
                    ax.annotate(dataset.upper(), (intrinsic_dims[i], complexities[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                ax.set_xlabel('Avg Intrinsic Dimension')
                ax.set_ylabel('Avg Feature Complexity')
                ax.set_title(f'Feature Dynamics - {model}')
                ax.grid(True, alpha=0.3)
            
            # Summary
            ax = axes[1, 1]
            ax.axis('off')
            
            summary_text = f"Model: {model}\nDatasets: {len(datasets)}\n\n"
            for dataset in datasets:
                summary_text += f"{dataset.upper()}:\n"
                if 'macroscopic' in model_results[model][dataset]:
                    summary = model_results[model][dataset]['macroscopic'].get('information_flow', {}).get('summary', {})
                    summary_text += f"  Compression: {summary.get('total_compression', 0):.3f}\n"
                    summary_text += f"  Task Info: {summary.get('total_task_info_gain', 0):.3f}\n"
                summary_text += "\n"
            
            ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace')
            
            plt.suptitle(f'Dataset Comparison - {model}', fontsize=16)
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{model}_dataset_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    def plot_architecture_comparison(self, all_results: Dict):
        """Plot comparison across architecture types"""
        # Group by architecture type
        arch_results = {'mlp': {}, 'cnn': {}, 'resnet': {}, 'vit': {}}
        
        for dataset in all_results:
            for model in all_results[dataset]:
                for arch_type, model_list in self.supported_models.items():
                    if any(m in model.lower() for m in model_list):
                        if dataset not in arch_results[arch_type]:
                            arch_results[arch_type][dataset] = {}
                        arch_results[arch_type][dataset][model] = all_results[dataset][model]
                        break
        
        # Create architecture comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['compression', 'task_info', 'ntk_rank', 'intrinsic_dim']
        arch_types = list(arch_results.keys())
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            
            # Collect data for each architecture
            arch_data = []
            for arch_type in arch_types:
                if not arch_results[arch_type]:
                    continue
                    
                values = []
                for dataset in arch_results[arch_type]:
                    for model in arch_results[arch_type][dataset]:
                        if metric == 'compression':
                            summary = arch_results[arch_type][dataset][model].get('macroscopic', {}).get('information_flow', {}).get('summary', {})
                            values.append(summary.get('total_compression', 0))
                        elif metric == 'task_info':
                            summary = arch_results[arch_type][dataset][model].get('macroscopic', {}).get('information_flow', {}).get('summary', {})
                            values.append(summary.get('total_task_info_gain', 0))
                        elif metric == 'ntk_rank':
                            ntk_evo = arch_results[arch_type][dataset][model].get('mesoscopic', {}).get('ntk', {}).get('evolution', {})
                            values.append(ntk_evo.get('rank_compression', 0))
                        elif metric == 'intrinsic_dim':
                            dynamics = arch_results[arch_type][dataset][model].get('mesoscopic', {}).get('dynamics', {})
                            values.append(np.mean(dynamics.get('intrinsic_dimension', [0])))
                
                if values:
                    arch_data.append(values)
            
            # Create box plot
            if arch_data:
                ax.boxplot(arch_data, labels=[arch.upper() for arch in arch_types[:len(arch_data)]])
                ax.set_title(f'{metric.replace("_", " ").title()}')
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Architecture Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'architecture_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _finalize_experiment_log(self):
        """Finalize experiment logging and save summary"""
        self.experiment_log['end_time'] = datetime.now().isoformat()
        self.experiment_log['analysis_duration'] = time.time() - time.mktime(
            datetime.fromisoformat(self.experiment_log['start_time']).timetuple()
        )
        
        # Save experiment log
        log_path = self.output_dir / f"experiment_log_{self.experiment_id}.json"
        with open(log_path, 'w') as f:
            json.dump(self.experiment_log, f, indent=2, cls=NumpyEncoder)
        
        print(f"\n=== Experiment Summary ===")
        print(f"Experiment ID: {self.experiment_id}")
        print(f"Device: {self.experiment_log['device']}")
        print(f"Models analyzed: {len(self.experiment_log['models_analyzed'])}")
        print(f"Datasets analyzed: {len(self.experiment_log['datasets_analyzed'])}")
        print(f"Total samples processed: {self.experiment_log['total_samples_processed']}")
        print(f"Total duration: {self.experiment_log['analysis_duration']:.2f} seconds")
        print(f"Experiment log saved to: {log_path}")
        
        if self.writer:
            print(f"TensorBoard logs: {self.tensorboard_dir}")
            print(f"Run: tensorboard --logdir={self.tensorboard_dir}")
    
    def __del__(self):
        """Cleanup when analyzer is destroyed"""
        if hasattr(self, 'writer') and self.writer:
            self.writer.close()

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Multi-Model Analysis across architectures and datasets")
    
    # Model selection
    parser.add_argument('--models', nargs='+', 
                       default=['resnet18', 'vit_base_patch16_224', 'convnext_tiny', 'mlp_mixer_b16_224'],
                       help='List of models to analyze')
    
    # Dataset selection
    parser.add_argument('--datasets', nargs='+', 
                       default=['cifar10', 'cifar100', 'svhn'],
                       help='List of datasets to use')
    
    # Analysis options
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained models')
    
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (auto-detected if not specified)')
    
    parser.add_argument('--output_dir', type=str, default='./results/multi_model_analysis/',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = MultiModelAnalyzer(output_dir=args.output_dir)
    
    # Check model availability
    available_models = []
    for model in args.models:
        if model in timm.list_models():
            available_models.append(model)
        else:
            print(f"Warning: Model {model} not found in timm")
    
    if not available_models:
        print("No valid models found. Exiting.")
        return
    
    print(f"Analyzing {len(available_models)} models: {available_models}")
    print(f"Datasets: {args.datasets}")
    
    # Run analysis
    results = analyzer.run_comprehensive_analysis(
        models=available_models,
        datasets=args.datasets,
        pretrained=args.pretrained,
        device=args.device
    )
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")
    return results

if __name__ == "__main__":
    main()
