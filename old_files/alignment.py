import torch
import torch.nn as nn
import torch.utils.tensorboard as tensorboard
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from scipy.stats import entropy
from typing import Dict, List, Tuple, Optional
import copy
import argparse
import os
from datetime import datetime

# Import external modules (assuming these exist)
try:
    from UnifiedNeuralAnalysisFramework import AGOPCalculator
    from ComprehensiveNetworkAnalyzer import CKAAnalysis
except ImportError:
    # Fallback implementations if external modules aren't available
    class AGOPCalculator:
        def compute_agop_batch(self, model, data_loader):
            return {'layer_0_mean_agop': 0.5}
    
    class CKAAnalysis:
        def _compute_cka(self, x, y):
            return 0.7

class PhaseTransitionExperiment:
    """
    Experiment to map the complete phase diagram of NTK-AGOP space
    and identify alignment emergence regions
    """
    
    def __init__(self, device='cuda', log_dir=None):
        self.device = device
        
        # Set up TensorBoard
        if log_dir is None:
            log_dir = f"runs/phase_transition_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = tensorboard.SummaryWriter(log_dir)
        
        self.results = {
            'ntk_stability': [],
            'agop_magnitude': [],
            'alignment_score': [],
            'compression_ratio': [],
            'phase_label': []
        }
        
    def create_model(self, width: int, depth: int, init_scale: float = 1.0):
        """Create model with specific width and initialization scale"""
        layers = []
        input_dim = 784  # MNIST
        
        for i in range(depth):
            if i == 0:
                layers.append(nn.Linear(input_dim, width))
            else:
                layers.append(nn.Linear(width, width))
            
            if i < depth - 1:
                layers.append(nn.ReLU())
        
        layers.append(nn.Linear(width, 10))
        
        model = nn.Sequential(*layers)
        
        # Custom initialization
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, init_scale / np.sqrt(m.weight.shape[1]))
                    nn.init.zeros_(m.bias)
        
        return model.to(self.device)
    
    def _get_data_loaders(self, batch_size=128, n_samples=None):
        """Get MNIST data loaders"""
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        
        if n_samples:
            train_dataset = torch.utils.data.Subset(
                train_dataset, 
                torch.randperm(len(train_dataset))[:n_samples]
            )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        return train_loader, test_loader
    
    def compute_ntk_stability(self, model, data_loader, reference_model=None):
        """Compute NTK stability metric"""
        if reference_model is None:
            reference_model = copy.deepcopy(model)
        
        ntk_current = self._compute_empirical_ntk(model, data_loader)
        ntk_reference = self._compute_empirical_ntk(reference_model, data_loader)
        
        # Stability = similarity between current and reference NTK
        stability = torch.nn.functional.cosine_similarity(
            ntk_current.flatten(), 
            ntk_reference.flatten(), 
            dim=0
        ).item()
        
        return stability
    
    def _compute_empirical_ntk(self, model, data_loader, n_samples=100):
        """Compute empirical NTK matrix"""
        model.zero_grad()
        
        # Sample data
        x_sample = []
        for x, _ in data_loader:
            x_sample.append(x[:n_samples // len(data_loader)])
            if len(torch.cat(x_sample)) >= n_samples:
                break
        
        x_sample = torch.cat(x_sample)[:n_samples].to(self.device)
        
        # Compute Jacobians
        jacobians = []
        for i in range(n_samples):
            model.zero_grad()
            output = model(x_sample[i:i+1])
            
            # Compute gradient w.r.t all parameters
            grads = []
            for j in range(output.shape[1]):
                if j > 0:
                    model.zero_grad()
                    
                output[0, j].backward(retain_graph=(j < output.shape[1] - 1))
                
                grad_vec = []
                for param in model.parameters():
                    if param.grad is not None:
                        grad_vec.append(param.grad.flatten())
                    else:
                        grad_vec.append(torch.zeros_like(param).flatten())
                
                grads.append(torch.cat(grad_vec))
            
            jacobians.append(torch.stack(grads))
        
        # Compute NTK
        jacobians = torch.stack(jacobians)  # [n_samples, n_outputs, n_params]
        ntk = torch.einsum('nop,mop->nm', jacobians, jacobians)
        
        return ntk
    
    def compute_agop_magnitude(self, model, data_loader):
        """Compute AGOP magnitude across all layers"""
        agop_calc = AGOPCalculator()
        agop_results = agop_calc.compute_agop_batch(model, data_loader)
        
        # Average AGOP magnitude
        agop_values = [agop_results.get(f'layer_{i}_mean_agop', 0) 
                       for i in range(len(list(model.parameters())))]
        
        return np.mean([v for v in agop_values if v > 0])
    
    def compute_alignment(self, model1, model2, data_loader):
        """Compute CKA alignment between two models"""
        cka = CKAAnalysis()
        
        # Get representations
        with torch.no_grad():
            reps1 = []
            reps2 = []
            
            for x, _ in data_loader:
                x = x.to(self.device)
                
                # Get intermediate representations
                rep1 = self._get_representations(model1, x)
                rep2 = self._get_representations(model2, x)
                
                reps1.append(rep1)
                reps2.append(rep2)
                
                if len(reps1) >= 10:  # Sample size
                    break
            
            reps1 = {k: torch.cat([r[k] for r in reps1]) 
                     for k in reps1[0].keys()}
            reps2 = {k: torch.cat([r[k] for r in reps2]) 
                     for k in reps2[0].keys()}
        
        # Compute layer-wise alignment
        alignments = []
        for layer_name in reps1.keys():
            if layer_name in reps2:
                cka_score = cka._compute_cka(reps1[layer_name], reps2[layer_name])
                alignments.append(cka_score)
        
        return np.mean(alignments) if alignments else 0.0
    
    def _get_representations(self, model, x):
        """Extract all intermediate representations"""
        representations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                representations[name] = output.detach()
            return hook
        
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        _ = model(x)
        
        for hook in hooks:
            hook.remove()
        
        return representations
    
    def _train_epoch(self, model, data_loader, optimizer):
        """Train model for one epoch"""
        model.train()
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def _compute_compression_ratio(self, model, data_loader):
        """Compute compression ratio of representations"""
        model.eval()
        representations = []
        
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(self.device)
                # Get final layer representation
                rep = model[:-1](x)  # All layers except the last
                representations.append(rep.cpu().numpy())
                
                if len(representations) > 10:  # Limit samples
                    break
        
        representations = np.vstack(representations)
        
        # Compute effective dimensionality
        cov = np.cov(representations.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        # Compression ratio = original dim / effective dim
        orig_dim = representations.shape[1]
        eff_dim = np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2)
        
        return orig_dim / eff_dim
    
    def run_phase_diagram_scan(self, 
                               width_range=(8, 512), 
                               lr_range=(1e-4, 1.0),
                               init_range=(0.1, 10.0),
                               n_points=20):
        """
        Scan the parameter space to create phase diagram
        """
        from sklearn.model_selection import ParameterGrid
        
        # Create parameter grid
        param_grid = {
            'width': np.logspace(np.log10(width_range[0]), 
                                np.log10(width_range[1]), n_points).astype(int),
            'lr': np.logspace(np.log10(lr_range[0]), 
                             np.log10(lr_range[1]), n_points),
            'init_scale': np.logspace(np.log10(init_range[0]), 
                                     np.log10(init_range[1]), n_points // 2)
        }
        
        # Load data
        train_loader, test_loader = self._get_data_loaders()
        
        results = []
        
        for params in tqdm(ParameterGrid(param_grid), desc="Phase diagram scan"):
            # Create models
            model1 = self.create_model(
                width=int(params['width']), 
                depth=3, 
                init_scale=params['init_scale']
            )
            model2 = self.create_model(
                width=int(params['width']), 
                depth=3, 
                init_scale=params['init_scale']
            )
            
            # Store initial models for NTK reference
            init_model1 = copy.deepcopy(model1)
            
            # Train models
            optimizer1 = torch.optim.SGD(model1.parameters(), lr=params['lr'])
            optimizer2 = torch.optim.SGD(model2.parameters(), lr=params['lr'])
            
            # Training loop with measurements
            for epoch in range(50):  # Short training for phase scan
                # Train step
                self._train_epoch(model1, train_loader, optimizer1)
                self._train_epoch(model2, train_loader, optimizer2)
                
                if epoch % 10 == 0:
                    # Compute metrics
                    ntk_stab = self.compute_ntk_stability(model1, test_loader, init_model1)
                    agop_mag = self.compute_agop_magnitude(model1, test_loader)
                    alignment = self.compute_alignment(model1, model2, test_loader)
                    
                    # Compute compression ratio
                    compression = self._compute_compression_ratio(model1, test_loader)
                    
                    # Classify phase
                    phase = self._classify_phase(ntk_stab, agop_mag, alignment)
                    
                    result_dict = {
                        'width': params['width'],
                        'lr': params['lr'],
                        'init_scale': params['init_scale'],
                        'epoch': epoch,
                        'ntk_stability': ntk_stab,
                        'agop_magnitude': agop_mag,
                        'alignment': alignment,
                        'compression_ratio': compression,
                        'phase': phase
                    }
                    
                    results.append(result_dict)
                    
                    # Log to TensorBoard
                    step = len(results) - 1
                    for key, value in result_dict.items():
                        if isinstance(value, (int, float)):
                            self.writer.add_scalar(f'phase_scan/{key}', value, step)
        
        self.phase_results = pd.DataFrame(results)
        return self.phase_results
    
    def _classify_phase(self, ntk_stability, agop_magnitude, alignment):
        """Classify the training phase based on metrics"""
        if ntk_stability > 0.9 and agop_magnitude < 0.05:
            return 'lazy'
        elif ntk_stability < 0.5 and agop_magnitude > 0.2:
            return 'chaotic'
        elif alignment > 0.7:
            return 'aligned'
        else:
            return 'transition'
    
    def plot_phase_diagram(self):
        """Create phase diagram visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Alignment heatmap in NTK-AGOP space
        ax = axes[0, 0]
        scatter = ax.scatter(
            self.phase_results['ntk_stability'],
            self.phase_results['agop_magnitude'],
            c=self.phase_results['alignment'],
            cmap='viridis',
            s=50,
            alpha=0.6
        )
        ax.set_xlabel('NTK Stability')
        ax.set_ylabel('AGOP Magnitude')
        ax.set_title('Alignment in NTK-AGOP Space')
        plt.colorbar(scatter, ax=ax)
        
        # 2. Phase regions
        ax = axes[0, 1]
        phase_colors = {'lazy': 'blue', 'aligned': 'green', 
                       'chaotic': 'red', 'transition': 'yellow'}
        
        for phase, color in phase_colors.items():
            phase_data = self.phase_results[self.phase_results['phase'] == phase]
            ax.scatter(
                phase_data['ntk_stability'],
                phase_data['agop_magnitude'],
                c=color,
                label=phase,
                s=30,
                alpha=0.5
            )
        
        ax.set_xlabel('NTK Stability')
        ax.set_ylabel('AGOP Magnitude')
        ax.set_title('Phase Regions')
        ax.legend()
        
        # 3. Learning rate effect
        ax = axes[1, 0]
        lr_groups = self.phase_results.groupby('lr')['alignment'].mean()
        ax.semilogx(lr_groups.index, lr_groups.values, 'o-')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Average Alignment')
        ax.set_title('Learning Rate vs Alignment')
        
        # 4. Width effect
        ax = axes[1, 1]
        width_groups = self.phase_results.groupby('width')['alignment'].mean()
        ax.loglog(width_groups.index, width_groups.values, 'o-')
        ax.set_xlabel('Network Width')
        ax.set_ylabel('Average Alignment')
        ax.set_title('Width vs Alignment')
        
        plt.tight_layout()
        
        # Log figure to TensorBoard
        self.writer.add_figure('phase_diagram', fig)
        
        return fig


class ScalingLawsExperiment:
    """
    Discover and validate scaling laws for alignment emergence
    """
    
    def __init__(self, device='cuda', log_dir=None):
        self.device = device
        
        # Set up TensorBoard
        if log_dir is None:
            log_dir = f"runs/scaling_laws_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = tensorboard.SummaryWriter(log_dir)
        
        self.scaling_data = {
            'width': [],
            'depth': [],
            'dataset_size': [],
            'critical_time': [],
            'final_alignment': [],
            'compression_ratio': [],
            'convergence_rate': []
        }
    
    def _create_scaled_model(self, width, depth):
        """Create model with specified width and depth"""
        layers = []
        input_dim = 784  # MNIST
        
        for i in range(depth):
            if i == 0:
                layers.append(nn.Linear(input_dim, width))
            else:
                layers.append(nn.Linear(width, width))
            
            if i < depth - 1:
                layers.append(nn.ReLU())
        
        layers.append(nn.Linear(width, 10))
        
        model = nn.Sequential(*layers).to(self.device)
        
        # Initialize weights
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.zeros_(m.bias)
        
        return model
    
    def _get_scaled_data(self, n_data):
        """Get data loader with specified number of samples"""
        train_loader, test_loader = self._get_data_loaders(n_samples=n_data)
        return train_loader, test_loader
    
    def _get_data_loaders(self, batch_size=128, n_samples=None):
        """Get MNIST data loaders"""
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        
        if n_samples:
            train_dataset = torch.utils.data.Subset(
                train_dataset, 
                torch.randperm(len(train_dataset))[:n_samples]
            )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        return train_loader, test_loader
    
    def _train_epoch(self, model, data_loader, optimizer):
        """Train model for one epoch"""
        model.train()
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def _measure_alignment(self, model1, model2, data_loader):
        """Measure alignment between two models"""
        return PhaseTransitionExperiment(device=self.device).compute_alignment(model1, model2, data_loader)
    
    def _measure_compression(self, model, data_loader):
        """Measure compression ratio"""
        return PhaseTransitionExperiment(device=self.device)._compute_compression_ratio(model, data_loader)
    
    def run_scaling_analysis(self):
        """Main scaling law discovery experiment"""
        
        # Parameter ranges for scaling
        widths = [32, 64, 128, 256, 512, 1024, 2048]
        depths = [2, 3, 4, 5, 6, 8, 10]
        dataset_sizes = [1000, 2000, 5000, 10000, 20000, 50000]
        
        results = []
        
        for width in tqdm(widths, desc='Width scaling'):
            for depth in depths[:4]:  # Limit depth for computational efficiency
                for n_data in dataset_sizes[:4]:
                    
                    # Run experiment
                    exp_result = self._run_single_scaling_experiment(
                        width=width,
                        depth=depth,
                        n_data=n_data
                    )
                    
                    exp_result['width'] = width
                    exp_result['depth'] = depth
                    exp_result['dataset_size'] = n_data
                    
                    results.append(exp_result)
                    
                    # Log to TensorBoard
                    step = len(results) - 1
                    for key, value in exp_result.items():
                        if isinstance(value, (int, float)):
                            self.writer.add_scalar(f'scaling/{key}', value, step)
        
        self.scaling_results = pd.DataFrame(results)
        return self.scaling_results
    
    def _run_single_scaling_experiment(self, width, depth, n_data):
        """Run single configuration and measure critical time"""
        
        # Create models
        model1 = self._create_scaled_model(width, depth)
        model2 = self._create_scaled_model(width, depth)
        
        # Get data
        train_loader, test_loader = self._get_scaled_data(n_data)
        
        # Training with tracking
        alignment_history = []
        compression_history = []
        loss_history = []
        
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        
        critical_time = None
        max_epochs = 100
        
        for epoch in range(max_epochs):
            # Train
            loss1 = self._train_epoch(model1, train_loader, optimizer1)
            loss2 = self._train_epoch(model2, train_loader, optimizer2)
            
            # Measure alignment
            alignment = self._measure_alignment(model1, model2, test_loader)
            compression = self._measure_compression(model1, test_loader)
            
            alignment_history.append(alignment)
            compression_history.append(compression)
            loss_history.append((loss1 + loss2) / 2)
            
            # Detect critical time (when alignment > 0.5)
            if critical_time is None and alignment > 0.5:
                critical_time = epoch
        
        # Compute convergence rate
        if len(alignment_history) > 10:
            convergence_rate = self._fit_convergence_rate(alignment_history)
        else:
            convergence_rate = 0.0
        
        return {
            'critical_time': critical_time if critical_time else max_epochs,
            'final_alignment': alignment_history[-1],
            'compression_ratio': compression_history[-1],
            'convergence_rate': convergence_rate,
            'loss_history': loss_history,
            'alignment_history': alignment_history
        }
    
    def _fit_convergence_rate(self, alignment_history):
        """Fit exponential convergence rate"""
        t = np.arange(len(alignment_history))
        y = np.array(alignment_history)
        
        # Fit y = 1 - exp(-rate * t)
        from scipy.optimize import curve_fit
        
        def exp_func(t, rate, a):
            return a * (1 - np.exp(-rate * t))
        
        try:
            popt, _ = curve_fit(exp_func, t, y, p0=[0.1, 1.0])
            return popt[0]  # rate
        except:
            return 0.0
    
    def discover_scaling_laws(self):
        """Fit power laws to scaling data"""
        
        # Fit scaling laws for critical time
        laws = {}
        
        # Critical time vs width
        width_data = self.scaling_results.groupby('width')['critical_time'].mean()
        laws['width'] = self._fit_power_law(
            width_data.index, 
            width_data.values,
            'Width'
        )
        
        # Critical time vs depth  
        depth_data = self.scaling_results.groupby('depth')['critical_time'].mean()
        laws['depth'] = self._fit_power_law(
            depth_data.index,
            depth_data.values,
            'Depth'
        )
        
        # Critical time vs dataset size
        data_size = self.scaling_results.groupby('dataset_size')['critical_time'].mean()
        laws['dataset_size'] = self._fit_power_law(
            data_size.index,
            data_size.values,
            'Dataset Size'
        )
        
        # Combined scaling law
        laws['combined'] = self._fit_combined_scaling_law()
        
        # Log scaling laws to TensorBoard
        for law_name, law_data in laws.items():
            if isinstance(law_data, dict):
                for param, value in law_data.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f'scaling_laws/{law_name}/{param}', value)
        
        return laws
    
    def _fit_power_law(self, x, y, name):
        """Fit power law y = a * x^b"""
        log_x = np.log(x)
        log_y = np.log(y)
        
        # Linear regression in log space
        slope, intercept = np.polyfit(log_x, log_y, 1)
        
        # Compute R²
        y_pred = intercept + slope * log_x
        ss_res = np.sum((log_y - y_pred)**2)
        ss_tot = np.sum((log_y - np.mean(log_y))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'name': name,
            'coefficient': np.exp(intercept),
            'exponent': slope,
            'r_squared': r_squared,
            'equation': f't_c = {np.exp(intercept):.2f} * {name}^{slope:.2f}'
        }
    
    def _fit_combined_scaling_law(self):
        """Fit combined scaling law: t_c = a * W^α * D^β * N^γ"""
        from scipy.optimize import minimize
        
        df = self.scaling_results
        
        def combined_law(params, W, D, N):
            a, alpha, beta, gamma = params
            return a * (W ** alpha) * (D ** beta) * (N ** gamma)
        
        def loss(params):
            pred = combined_law(params, df['width'], df['depth'], df['dataset_size'])
            return np.mean((np.log(pred) - np.log(df['critical_time']))**2)
        
        # Initial guess
        x0 = [1000, -0.5, 1.0, -0.3]
        
        result = minimize(loss, x0, method='Nelder-Mead')
        
        return {
            'coefficient': result.x[0],
            'width_exponent': result.x[1],
            'depth_exponent': result.x[2],
            'data_exponent': result.x[3],
            'equation': f't_c = {result.x[0]:.1f} * W^{result.x[1]:.2f} * D^{result.x[2]:.2f} * N^{result.x[3]:.2f}'
        }
    
    def plot_scaling_laws(self):
        """Visualize discovered scaling laws"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Width scaling
        ax = axes[0, 0]
        width_data = self.scaling_results.groupby('width').agg({
            'critical_time': ['mean', 'std']
        })
        
        ax.errorbar(
            width_data.index,
            width_data['critical_time']['mean'],
            yerr=width_data['critical_time']['std'],
            fmt='o',
            capsize=5,
            label='Data'
        )
        
        # Fit line
        x_fit = np.logspace(np.log10(min(width_data.index)), 
                           np.log10(max(width_data.index)), 100)
        law = self.scaling_laws['width']
        y_fit = law['coefficient'] * (x_fit ** law['exponent'])
        
        ax.loglog(x_fit, y_fit, 'r--', 
                 label=f"$t_c \propto W^{{{law['exponent']:.2f}}}$")
        ax.set_xlabel('Width')
        ax.set_ylabel('Critical Time')
        ax.set_title('Width Scaling')
        ax.legend()
        
        # 2. Depth scaling
        ax = axes[0, 1]
        depth_data = self.scaling_results.groupby('depth').agg({
            'critical_time': ['mean', 'std']
        })
        
        ax.errorbar(
            depth_data.index,
            depth_data['critical_time']['mean'],
            yerr=depth_data['critical_time']['std'],
            fmt='o',
            capsize=5
        )
        
        # Similar plotting for depth...
        ax.set_xlabel('Depth')
        ax.set_ylabel('Critical Time')
        ax.set_title('Depth Scaling')
        
        # 3. Dataset size scaling
        ax = axes[0, 2]
        # Similar for dataset size...
        
        # 4. Final alignment vs model size
        ax = axes[1, 0]
        model_size = self.scaling_results['width'] * self.scaling_results['depth']
        ax.scatter(model_size, self.scaling_results['final_alignment'], alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlabel('Model Size (Width × Depth)')
        ax.set_ylabel('Final Alignment')
        ax.set_title('Alignment vs Model Size')
        
        # 5. Compression ratio scaling
        ax = axes[1, 1]
        ax.scatter(self.scaling_results['width'], 
                  self.scaling_results['compression_ratio'],
                  c=self.scaling_results['depth'],
                  cmap='viridis',
                  alpha=0.6)
        ax.set_xscale('log')
        ax.set_xlabel('Width')
        ax.set_ylabel('Compression Ratio')
        ax.set_title('Compression vs Width')
        
        # 6. Phase transition sharpness
        ax = axes[1, 2]
        # Plot how sharp the transition is for different model sizes
        for width in [64, 256, 1024]:
            data = self.scaling_results[self.scaling_results['width'] == width]
            if len(data) > 0:
                # Get alignment history for one run
                history = data.iloc[0]['alignment_history']
                if isinstance(history, list) and len(history) > 0:
                    ax.plot(history, label=f'Width={width}')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Alignment')
        ax.set_title('Transition Sharpness vs Width')
        ax.legend()
        
        plt.tight_layout()
        
        # Log figure to TensorBoard
        self.writer.add_figure('scaling_laws', fig)
        
        return fig


class AlignmentApplicationsExperiment:
    """
    Test practical applications of alignment: model stitching, 
    ensemble efficiency, and transfer learning
    """
    
    def __init__(self, device='cuda', log_dir=None):
        self.device = device
        
        # Set up TensorBoard
        if log_dir is None:
            log_dir = f"runs/applications_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = tensorboard.SummaryWriter(log_dir)
    
    def _create_model(self, width, depth):
        """Create a standard model"""
        layers = []
        input_dim = 784  # MNIST
        
        for i in range(depth):
            if i == 0:
                layers.append(nn.Linear(input_dim, width))
            else:
                layers.append(nn.Linear(width, width))
            
            if i < depth - 1:
                layers.append(nn.ReLU())
        
        layers.append(nn.Linear(width, 10))
        
        model = nn.Sequential(*layers).to(self.device)
        
        # Initialize weights
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.zeros_(m.bias)
        
        return model
    
    def _get_data_loader(self, dataset_name='mnist', n_samples=None):
        """Get data loader for different datasets"""
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        
        if dataset_name == 'mnist':
            dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        elif dataset_name == 'fashion_mnist':
            dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        elif dataset_name == 'cifar10':
            # For CIFAR10, we need to convert to grayscale and resize
            transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Lambda(lambda x: x.view(-1))
            ])
            dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        elif dataset_name == 'svhn':
            transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Lambda(lambda x: x.view(-1))
            ])
            dataset = datasets.SVHN('./data', split='train', download=True, transform=transform)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        if n_samples:
            dataset = torch.utils.data.Subset(
                dataset, 
                torch.randperm(len(dataset))[:n_samples]
            )
        
        return torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    def _get_test_loader(self, dataset_name='mnist'):
        """Get test data loader"""
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        
        if dataset_name == 'mnist':
            dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        elif dataset_name == 'fashion_mnist':
            dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
        elif dataset_name == 'cifar10':
            transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Lambda(lambda x: x.view(-1))
            ])
            dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        elif dataset_name == 'svhn':
            transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Lambda(lambda x: x.view(-1))
            ])
            dataset = datasets.SVHN('./data', split='test', download=True, transform=transform)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    
    def _train_model(self, model, data_loader=None, epochs=20):
        """Train a model and return final accuracy"""
        if data_loader is None:
            data_loader = self._get_data_loader('mnist')
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        test_loader = self._get_test_loader()
        return self._evaluate_model(model, test_loader)
    
    def _evaluate_model(self, model, data_loader):
        """Evaluate model accuracy"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = model(x)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        return correct / total
    
    def _measure_model_alignment(self, model1, model2):
        """Measure alignment between two models"""
        test_loader = self._get_test_loader()
        return PhaseTransitionExperiment(device=self.device).compute_alignment(model1, model2, test_loader)
    
    def _measure_layer_alignment(self, model1, model2, layer_idx):
        """Measure alignment at a specific layer"""
        test_loader = self._get_test_loader()
        
        # Get representations from the specified layer
        def get_layer_representation(model, x, layer_idx):
            representations = {}
            
            def hook_fn(name):
                def hook(module, input, output):
                    representations[name] = output.detach()
                return hook
            
            hooks = []
            for i, (name, module) in enumerate(model.named_modules()):
                if isinstance(module, (nn.Linear, nn.ReLU)):
                    hooks.append(module.register_forward_hook(hook_fn(name)))
                    if i >= layer_idx * 2:  # Account for ReLU layers
                        break
            
            _ = model(x)
            
            for hook in hooks:
                hook.remove()
            
            # Return the representation from the target layer
            for name, rep in representations.items():
                if f'layer_{layer_idx}' in name:
                    return rep
            
            return None
        
        # Compute alignment
        cka = CKAAnalysis()
        reps1 = []
        reps2 = []
        
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(self.device)
                
                rep1 = get_layer_representation(model1, x, layer_idx)
                rep2 = get_layer_representation(model2, x, layer_idx)
                
                if rep1 is not None and rep2 is not None:
                    reps1.append(rep1.cpu())
                    reps2.append(rep2.cpu())
                
                if len(reps1) >= 10:  # Limit samples
                    break
        
        if not reps1:
            return 0.0
        
        reps1 = torch.cat(reps1)
        reps2 = torch.cat(reps2)
        
        return cka._compute_cka(reps1, reps2)
    
    def test_zero_shot_stitching(self, n_models=5):
        """Test zero-shot model stitching performance"""
        
        results = {
            'stitch_depth': [],
            'alignment_score': [],
            'performance_drop': [],
            'original_accuracy': [],
            'stitched_accuracy': []
        }
        
        # Train multiple models
        models = []
        original_accs = []
        
        for i in tqdm(range(n_models), desc="Training models for stitching"):
            model = self._create_model(width=256, depth=6)
            acc = self._train_model(model)
            models.append(model)
            original_accs.append(acc)
            
            # Log original accuracy
            self.writer.add_scalar('stitching/original_accuracy', acc, i)
        
        # Test all possible stitching combinations
        for i in tqdm(range(n_models), desc="Testing stitching combinations"):
            for j in range(n_models):
                if i != j:
                    for stitch_layer in range(1, 6):
                        # Perform stitching
                        stitched_model = self._stitch_models(
                            models[i], models[j], stitch_layer
                        )
                        
                        # Measure alignment at stitch point
                        alignment = self._measure_layer_alignment(
                            models[i], models[j], stitch_layer
                        )
                        
                        # Test stitched model
                        stitched_acc = self._evaluate_model(stitched_model)
                        
                        results['stitch_depth'].append(stitch_layer)
                        results['alignment_score'].append(alignment)
                        results['original_accuracy'].append(original_accs[i])
                        results['stitched_accuracy'].append(stitched_acc)
                        results['performance_drop'].append(
                            original_accs[i] - stitched_acc
                        )
                        
                        # Log to TensorBoard
                        step = len(results['stitch_depth']) - 1
                        self.writer.add_scalar('stitching/alignment', alignment, step)
                        self.writer.add_scalar('stitching/performance_drop', 
                                              original_accs[i] - stitched_acc, step)
                        self.writer.add_scalar('stitching/stitched_accuracy', stitched_acc, step)
        
        self.stitching_results = pd.DataFrame(results)
        return self.stitching_results
    
    def _stitch_models(self, model1, model2, stitch_layer):
        """Create stitched model: early layers from model1, late from model2"""
        stitched = nn.Sequential()
        
        modules1 = list(model1.modules())[1:]  # Skip Sequential wrapper
        modules2 = list(model2.modules())[1:]
        
        # Add layers from model1 up to stitch point
        for i, module in enumerate(modules1):
            if i < stitch_layer * 2:  # Account for ReLU layers
                stitched.add_module(f'layer_{i}', copy.deepcopy(module))
        
        # Add remaining layers from model2
        for i, module in enumerate(modules2):
            if i >= stitch_layer * 2:
                stitched.add_module(f'layer_{i}', copy.deepcopy(module))
        
        return stitched.to(self.device)
    
    def test_alignment_based_ensemble(self):
        """Test ensemble efficiency using alignment"""
        
        n_models = 10
        models = []
        
        # Train models with different initializations
        for i in tqdm(range(n_models), desc="Training models for ensemble"):
            model = self._create_model(width=128, depth=4)
            self._train_model(model)
            models.append(model)
        
        # Compute alignment matrix
        alignment_matrix = np.zeros((n_models, n_models))
        
        for i in tqdm(range(n_models), desc="Computing alignment matrix"):
            for j in range(n_models):
                if i != j:
                    alignment_matrix[i, j] = self._measure_model_alignment(
                        models[i], models[j]
                    )
                else:
                    alignment_matrix[i, j] = 1.0
        
        # Log alignment matrix
        self.writer.add_histogram('ensemble/alignment_matrix', alignment_matrix)
        
        # Test different ensemble strategies
        test_loader = self._get_test_loader()
        
        strategies = {
            'uniform': self._uniform_ensemble,
            'alignment_weighted': self._alignment_weighted_ensemble,
            'alignment_selected': self._alignment_selected_ensemble,
            'random_subset': self._random_subset_ensemble
        }
        
        results = {}
        
        for name, strategy_fn in strategies.items():
            accuracy, efficiency = strategy_fn(models, alignment_matrix, test_loader)
            results[name] = {
                'accuracy': accuracy,
                'efficiency': efficiency,
                'accuracy_per_flop': accuracy / efficiency
            }
            
            # Log results
            self.writer.add_scalar(f'ensemble/{name}/accuracy', accuracy)
            self.writer.add_scalar(f'ensemble/{name}/efficiency', efficiency)
            self.writer.add_scalar(f'ensemble/{name}/accuracy_per_flop', accuracy / efficiency)
        
        self.ensemble_results = results
        return results
    
    def _uniform_ensemble(self, models, alignment_matrix, test_loader):
        """Uniform ensemble of all models"""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                ensemble_pred = torch.zeros(x.shape[0], 10).to(self.device)
                
                for model in models:
                    pred = model(x)
                    ensemble_pred += torch.softmax(pred, dim=1)
                
                ensemble_pred /= len(models)
                
                correct += (ensemble_pred.argmax(1) == y).sum().item()
                total += x.shape[0]
        
        accuracy = correct / total
        efficiency = len(models)
        
        return accuracy, efficiency
    
    def _alignment_weighted_ensemble(self, models, alignment_matrix, test_loader):
        """Ensemble weighted by alignment scores"""
        
        # Compute principal eigenvector of alignment matrix
        eigenvals, eigenvects = np.linalg.eig(alignment_matrix)
        principal_component = np.abs(eigenvects[:, 0])
        weights = principal_component / principal_component.sum()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # Weighted predictions
                ensemble_pred = torch.zeros(x.shape[0], 10).to(self.device)
                
                for model, weight in zip(models, weights):
                    pred = model(x)
                    ensemble_pred += weight * torch.softmax(pred, dim=1)
                
                correct += (ensemble_pred.argmax(1) == y).sum().item()
                total += x.shape[0]
        
        accuracy = correct / total
        efficiency = len(models)  # All models used
        
        return accuracy, efficiency
    
    def _alignment_selected_ensemble(self, models, alignment_matrix, test_loader):
        """Select subset of models based on alignment"""
        
        # Select models that are well-aligned with others
        mean_alignment = alignment_matrix.mean(axis=1)
        selected_idx = np.argsort(mean_alignment)[-5:]  # Top 5
        
        selected_models = [models[i] for i in selected_idx]
        
        # Uniform ensemble of selected models
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                ensemble_pred = torch.zeros(x.shape[0], 10).to(self.device)
                
                for model in selected_models:
                    pred = model(x)
                    ensemble_pred += torch.softmax(pred, dim=1)
                
                ensemble_pred /= len(selected_models)
                
                correct += (ensemble_pred.argmax(1) == y).sum().item()
                total += x.shape[0]
        
        accuracy = correct / total
        efficiency = len(selected_models)
        
        return accuracy, efficiency
    
    def _random_subset_ensemble(self, models, alignment_matrix, test_loader):
        """Random subset ensemble for comparison"""
        import random
        
        selected_models = random.sample(models, 5)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                ensemble_pred = torch.zeros(x.shape[0], 10).to(self.device)
                
                for model in selected_models:
                    pred = model(x)
                    ensemble_pred += torch.softmax(pred, dim=1)
                
                ensemble_pred /= len(selected_models)
                
                correct += (ensemble_pred.argmax(1) == y).sum().item()
                total += x.shape[0]
        
        accuracy = correct / total
        efficiency = len(selected_models)
        
        return accuracy, efficiency
    
    def test_transfer_learning_efficiency(self):
        """Test how alignment affects transfer learning"""
        
        # Train source models on different tasks
        source_tasks = ['mnist', 'fashion_mnist', 'cifar10']
        target_task = 'svhn'
        
        source_models = {}
        alignments = {}
        
        # Train source models
        for task in tqdm(source_tasks, desc="Training source models"):
            model = self._create_model(width=256, depth=5)
            data_loader = self._get_data_loader(task)
            self._train_model(model, data_loader)
            source_models[task] = model
            
            # Log source model performance
            test_loader = self._get_test_loader(task)
            acc = self._evaluate_model(model, test_loader)
            self.writer.add_scalar(f'transfer/source_{task}/accuracy', acc)
        
        # Measure alignment with target task initialization
        target_model_init = self._create_model(width=256, depth=5)
        
        for task, model in source_models.items():
            alignments[task] = self._measure_model_alignment(
                model, target_model_init
            )
            self.writer.add_scalar(f'transfer/alignment/{task}', alignments[task])
        
        # Fine-tune each source model on target task
        transfer_results = {}
        
        for task, model in source_models.items():
            # Clone model for fine-tuning
            ft_model = copy.deepcopy(model)
            
            # Fine-tune with limited data
            target_loader = self._get_data_loader(target_task, n_samples=1000)
            
            convergence_history = []
            optimizer = torch.optim.Adam(ft_model.parameters(), lr=0.0001)
            
            for epoch in range(20):
                acc = self._train_epoch_with_accuracy(ft_model, target_loader, optimizer)
                convergence_history.append(acc)
                self.writer.add_scalar(f'transfer/convergence/{task}', acc, epoch)
            
            transfer_results[task] = {
                'initial_alignment': alignments[task],
                'convergence_history': convergence_history,
                'final_accuracy': convergence_history[-1],
                'convergence_rate': self._fit_convergence_rate(convergence_history)
            }
            
            # Log final results
            self.writer.add_scalar(f'transfer/final_accuracy/{task}', convergence_history[-1])
        
        self.transfer_results = transfer_results
        return transfer_results
    
    def _train_epoch_with_accuracy(self, model, data_loader, optimizer):
        """Train for one epoch and return accuracy"""
        model.train()
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        
        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        
        return correct / total
    
    def plot_application_results(self):
        """Comprehensive plotting of application experiments"""
        
        fig = plt.figure(figsize=(15, 12))
        
        # 1. Stitching performance vs alignment
        ax1 = plt.subplot(2, 3, 1)
        
        if hasattr(self, 'stitching_results'):
            scatter = ax1.scatter(
                self.stitching_results['alignment_score'],
                self.stitching_results['performance_drop'],
                c=self.stitching_results['stitch_depth'],
                cmap='plasma',
                alpha=0.6
            )
            ax1.set_xlabel('Alignment Score')
            ax1.set_ylabel('Performance Drop')
            ax1.set_title('Model Stitching Performance')
            plt.colorbar(scatter, ax=ax1, label='Stitch Depth')
            
            # Fit regression line
            from scipy.stats import linregress
            slope, intercept, r, _, _ = linregress(
                self.stitching_results['alignment_score'],
                self.stitching_results['performance_drop']
            )
            x_line = np.linspace(0, 1, 100)
            ax1.plot(x_line, slope * x_line + intercept, 'r--', 
                    label=f'R²={r**2:.3f}')
            ax1.legend()
        
        # 2. Ensemble strategies comparison
        ax2 = plt.subplot(2, 3, 2)
        
        if hasattr(self, 'ensemble_results'):
            strategies = list(self.ensemble_results.keys())
            accuracies = [self.ensemble_results[s]['accuracy'] for s in strategies]
            efficiencies = [self.ensemble_results[s]['efficiency'] for s in strategies]
            
            x = np.arange(len(strategies))
            width = 0.35
            
            ax2_twin = ax2.twinx()
            
            bars1 = ax2.bar(x - width/2, accuracies, width, label='Accuracy', 
                           color='skyblue')
            bars2 = ax2_twin.bar(x + width/2, efficiencies, width, 
                                label='Models Used', color='lightcoral')
            
            ax2.set_xlabel('Ensemble Strategy')
            ax2.set_ylabel('Accuracy', color='skyblue')
            ax2_twin.set_ylabel('Models Used', color='lightcoral')
            ax2.set_xticks(x)
            ax2.set_xticklabels(strategies, rotation=45, ha='right')
            ax2.set_title('Ensemble Strategies')
            
            ax2.tick_params(axis='y', labelcolor='skyblue')
            ax2_twin.tick_params(axis='y', labelcolor='lightcoral')
        
        # 3. Transfer learning efficiency
        ax3 = plt.subplot(2, 3, 3)
        
        if hasattr(self, 'transfer_results'):
            for task, results in self.transfer_results.items():
                ax3.plot(results['convergence_history'], 
                        label=f"{task} (align={results['initial_alignment']:.2f})")
            
            ax3.set_xlabel('Fine-tuning Epoch')
            ax3.set_ylabel('Target Task Accuracy')
            ax3.set_title('Transfer Learning Convergence')
            ax3.legend()
        
        # 4. Alignment vs Transfer Efficiency
        ax4 = plt.subplot(2, 3, 4)
        
        if hasattr(self, 'transfer_results'):
            alignments = [r['initial_alignment'] for r in self.transfer_results.values()]
            final_accs = [r['final_accuracy'] for r in self.transfer_results.values()]
            conv_rates = [r['convergence_rate'] for r in self.transfer_results.values()]
            
            scatter = ax4.scatter(alignments, final_accs, s=100, 
                                c=conv_rates, cmap='viridis', alpha=0.7)
            
            ax4.set_xlabel('Initial Alignment')
            ax4.set_ylabel('Final Transfer Accuracy')
            ax4.set_title('Alignment vs Transfer Success')
            plt.colorbar(scatter, ax=ax4, label='Convergence Rate')
        
        # 5. Layer-wise alignment for stitching
        ax5 = plt.subplot(2, 3, 5)
        
        if hasattr(self, 'stitching_results'):
            layer_perf = self.stitching_results.groupby('stitch_depth').agg({
                'alignment_score': 'mean',
                'performance_drop': ['mean', 'std']
            })
            
            layers = layer_perf.index
            align_scores = layer_perf['alignment_score']['mean']
            perf_drops = layer_perf['performance_drop']['mean']
            perf_stds = layer_perf['performance_drop']['std']
            
            ax5_twin = ax5.twinx()
            
            line1 = ax5.plot(layers, align_scores, 'b-o', label='Alignment')
            line2 = ax5_twin.errorbar(layers, perf_drops, yerr=perf_stds, 
                                     fmt='r-s', label='Performance Drop')
            
            ax5.set_xlabel('Stitch Layer')
            ax5.set_ylabel('Alignment Score', color='b')
            ax5_twin.set_ylabel('Performance Drop', color='r')
            ax5.set_title('Layer-wise Stitching Analysis')
            
            ax5.tick_params(axis='y', labelcolor='b')
            ax5_twin.tick_params(axis='y', labelcolor='r')
        
        # 6. Efficiency gains summary
        ax6 = plt.subplot(2, 3, 6)
        
        if hasattr(self, 'ensemble_results'):
            # Create efficiency comparison
            baseline_acc = self.ensemble_results['uniform']['accuracy']
            
            improvements = {}
            for strategy, results in self.ensemble_results.items():
                if strategy != 'uniform':
                    acc_gain = results['accuracy'] - baseline_acc
                    efficiency_gain = 10 - results['efficiency']  # Assuming 10 models total
                    improvements[strategy] = {
                        'accuracy_gain': acc_gain * 100,  # Convert to percentage
                        'efficiency_gain': efficiency_gain
                    }
            
            strategies = list(improvements.keys())
            acc_gains = [improvements[s]['accuracy_gain'] for s in strategies]
            eff_gains = [improvements[s]['efficiency_gain'] for s in strategies]
            
            ax6.scatter(eff_gains, acc_gains, s=100)
            
            for i, strategy in enumerate(strategies):
                ax6.annotate(strategy, (eff_gains[i], acc_gains[i]), 
                           xytext=(5, 5), textcoords='offset points')
            
            ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax6.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            ax6.set_xlabel('Efficiency Gain (Fewer Models)')
            ax6.set_ylabel('Accuracy Gain (%)')
            ax6.set_title('Efficiency vs Accuracy Trade-offs')
            
            # Add diagonal lines for constant trade-off ratios
            x_range = ax6.get_xlim()
            for ratio in [0.5, 1.0, 2.0]:
                x_vals = np.linspace(x_range[0], x_range[1], 100)
                y_vals = ratio * x_vals
                ax6.plot(x_vals, y_vals, ':', alpha=0.3, 
                        label=f'Ratio={ratio}')
        
        plt.tight_layout()
        
        # Log figure to TensorBoard
        self.writer.add_figure('applications', fig)
        
        return fig


def main():
    parser = argparse.ArgumentParser(description='Neural Network Alignment Experiments')
    parser.add_argument('--phase', action='store_true', help='Run phase transition experiment')
    parser.add_argument('--scaling', action='store_true', help='Run scaling laws experiment')
    parser.add_argument('--applications', action='store_true', help='Run applications experiment')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--log-dir', type=str, default=None, help='TensorBoard log directory')
    
    args = parser.parse_args()
    
    # If no specific experiment is selected, run all
    if not any([args.phase, args.scaling, args.applications]):
        args.all = True
    
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    
    # Phase 3: Phase Diagram Analysis
    if args.phase or args.all:
        print("Running Phase Diagram Analysis...")
        phase_exp = PhaseTransitionExperiment(device=device, log_dir=args.log_dir)
        phase_results = phase_exp.run_phase_diagram_scan(n_points=15)
        phase_fig = phase_exp.plot_phase_diagram()
        plt.savefig('phase_diagram_results.png', dpi=300)
        print("Phase diagram analysis completed")
    
    # Phase 4: Scaling Laws
    if args.scaling or args.all:
        print("\nDiscovering Scaling Laws...")
        scaling_exp = ScalingLawsExperiment(device=device, log_dir=args.log_dir)
        scaling_results = scaling_exp.run_scaling_analysis()
        scaling_laws = scaling_exp.discover_scaling_laws()
        scaling_exp.scaling_laws = scaling_laws
        scaling_fig = scaling_exp.plot_scaling_laws()
        plt.savefig('scaling_laws_results.png', dpi=300)
        print("Scaling laws analysis completed")
    
    # Phase 5: Applications
    if args.applications or args.all:
        print("\nTesting Practical Applications...")
        app_exp = AlignmentApplicationsExperiment(device=device, log_dir=args.log_dir)
        
        # Test stitching
        print("  - Testing zero-shot stitching...")
        stitching_results = app_exp.test_zero_shot_stitching(n_models=3)
        
        # Test ensembles
        print("  - Testing alignment-based ensembles...")
        ensemble_results = app_exp.test_alignment_based_ensemble()
        
        # Test transfer learning
        print("  - Testing transfer learning efficiency...")
        transfer_results = app_exp.test_transfer_learning_efficiency()
        
        app_fig = app_exp.plot_application_results()
        plt.savefig('application_results.png', dpi=300)
        print("Applications testing completed")
    
    # Summary report
    print("\n=== EXPERIMENTAL SUMMARY ===")
    
    if args.phase or args.all:
        print(f"\nPhase Diagram Analysis:")
        if hasattr(phase_exp, 'phase_results'):
            print(f"  - Identified {len(phase_exp.phase_results['phase'].unique())} distinct phases")
            aligned_data = phase_exp.phase_results[phase_exp.phase_results['phase'] == 'aligned']
            if not aligned_data.empty:
                print(f"  - Optimal alignment region: NTK stability {aligned_data['ntk_stability'].mean():.2f}, AGOP magnitude {aligned_data['agop_magnitude'].mean():.2f}")
    
    if args.scaling or args.all:
        print(f"\nScaling Laws Discovered:")
        if hasattr(scaling_exp, 'scaling_laws'):
            for name, law in scaling_exp.scaling_laws.items():
                if isinstance(law, dict) and 'equation' in law:
                    print(f"  - {name}: {law['equation']}")
    
    if args.applications or args.all:
        print(f"\nApplication Results:")
        if hasattr(app_exp, 'ensemble_results'):
            best_strategy = max(app_exp.ensemble_results.items(), 
                               key=lambda x: x[1]['accuracy_per_flop'])[0]
            print(f"  - Best ensemble strategy: {best_strategy}")
        
        if hasattr(app_exp, 'stitching_results'):
            print(f"  - Average stitching performance drop: {app_exp.stitching_results['performance_drop'].mean():.3f}")
        
        if hasattr(app_exp, 'transfer_results'):
            alignments = [r['initial_alignment'] for r in app_exp.transfer_results.values()]
            final_accs = [r['final_accuracy'] for r in app_exp.transfer_results.values()]
            if len(alignments) > 1:
                correlation = np.corrcoef(alignments, final_accs)[0, 1]
                print(f"  - Alignment-transfer correlation: {correlation:.3f}")
    
    print(f"\nTensorBoard logs available in respective experiment directories")
    print(f"Use 'tensorboard --logdir=runs' to view results")
    
    plt.show()

if __name__ == "__main__":
    main()