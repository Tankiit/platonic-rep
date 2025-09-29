
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import numpy as np
from PIL import Image
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# --- Alignment Analyzer Class ---
class AlignmentAnalyzer:
    """Performs RWA, RGA, and GWA analyses."""

    def compute_all_alignments(self, model, data, layer_name, model_type='vision', n_samples=30, n_param_samples=10000):
        model.eval()

        # 1. Hc: Representation Covariance
        print("  Computing Representation Covariance (Hc)...")
        activations = self._get_activations(model, data, layer_name, model_type, n_samples)
        activations_flat = activations.reshape(activations.shape[0], -1)
        hc_matrix = np.cov(activations_flat, rowvar=False)
        hc_spectrum = self._get_normalized_eigenspectrum(hc_matrix)

        # 2. Zc: Weight Covariance
        print("  Computing Weight Covariance (Zc)...")
        weights = self._get_weights(model, layer_name, model_type)
        zc_matrix = np.cov(weights, rowvar=True)
        zc_spectrum = self._get_normalized_eigenspectrum(zc_matrix)

        # 3. Gc: Approximated Gradient Covariance
        print("  Computing Approximated Gradient Covariance (Gc)...")
        gradients = self._compute_model_gradients(model, data, model_type, n_samples)
        grad_flat = self._flatten_gradients(gradients)
        n_params = grad_flat.shape[1]
        if n_params > n_param_samples:
            param_indices = np.random.choice(n_params, n_param_samples, replace=False)
            grad_flat_sampled = grad_flat[:, param_indices]
        else:
            grad_flat_sampled = grad_flat
        gc_matrix = np.cov(grad_flat_sampled.T)
        gc_spectrum = self._get_normalized_eigenspectrum(gc_matrix)

        # 4. Compute Alignments
        print("  Computing alignments...")
        rwa_similarity = self._compute_spectrum_similarity(hc_spectrum, zc_spectrum)
        rga_similarity = self._compute_spectrum_similarity(hc_spectrum, gc_spectrum)
        gwa_similarity = self._compute_spectrum_similarity(gc_spectrum, zc_spectrum)

        return {
            'RWA': rwa_similarity,
            'RGA': rga_similarity,
            'GWA': gwa_similarity
        }

    def _get_activations(self, model, data, layer_name, model_type, n_samples):
        model.eval()
        with torch.no_grad():
            if model_type == 'vision':
                x = data[:n_samples]
                if layer_name == 'layer2':
                    x = model.conv1(x); x = model.bn1(x); x = model.relu(x); x = model.maxpool(x)
                    x = model.layer1(x); x = model.layer2(x)
                    return x.mean(dim=(2,3)).cpu().numpy()
                else: raise NotImplementedError
            elif model_type == 'language':
                input_ids = data['input_ids'][:n_samples]
                attention_mask = data['attention_mask'][:n_samples]
                extended_attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
                if layer_name == 'layer_2':
                    hidden_states = model.embeddings(input_ids)
                    for i in range(3):
                        hidden_states = model.transformer.layer[i](hidden_states, extended_attention_mask)[0]
                    return hidden_states.mean(dim=1).cpu().numpy()
                else: raise NotImplementedError
        return None

    def _get_weights(self, model, layer_name, model_type):
        model.eval()
        if model_type == 'vision':
            if layer_name == 'layer2':
                w = model.layer2[0].conv1.weight.detach().cpu().numpy()
                return w.reshape(w.shape[0], -1)
            else: raise NotImplementedError
        elif model_type == 'language':
            if layer_name == 'layer_2':
                w = model.transformer.layer[2].attention.out_lin.weight.detach().cpu().numpy()
                return w
            else: raise NotImplementedError
        return None

    def _compute_model_gradients(self, model, data, model_type, n_samples):
        model.train()
        gradients = []
        if model_type == 'language':
            input_ids = data['input_ids'][:n_samples]; attention_mask = data['attention_mask'][:n_samples]
            num_samples = len(input_ids)
        else:
            data = data[:n_samples]; num_samples = len(data)

        for i in range(num_samples):
            model.zero_grad()
            if model_type == 'language':
                output = model(input_ids=input_ids[i:i+1], attention_mask=attention_mask[i:i+1])
            else:
                output = model(data[i:i+1])
            if isinstance(output, dict): output = output['last_hidden_state']
            loss = torch.mean(output[0])
            loss.backward()
            grad_vec = []
            for param in model.parameters():
                if param.grad is not None: grad_vec.append(param.grad.view(-1).detach().cpu().numpy())
            if grad_vec: gradients.append(np.concatenate(grad_vec))
        return np.array(gradients)

    def _flatten_gradients(self, gradients):
        return gradients.reshape(gradients.shape[0], -1)

    def _get_normalized_eigenspectrum(self, matrix):
        eigenvals = np.linalg.eigvals(matrix)
        eigenvals = np.real(eigenvals[eigenvals > 1e-10])
        eigenvals = np.sort(eigenvals)[::-1]
        return eigenvals / np.sum(eigenvals)

    def _compute_spectrum_similarity(self, spec1, spec2):
        len1, len2 = len(spec1), len(spec2)
        if len1 > len2: spec2 = np.pad(spec2, (0, len1 - len2), 'constant')
        elif len2 > len1: spec1 = np.pad(spec1, (0, len2 - len1), 'constant')
        return np.dot(spec1, spec2) / (np.linalg.norm(spec1) * np.linalg.norm(spec2))

# --- Main Execution ---
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    N_SAMPLES = 30

    # 1. Load Models
    print("Loading models...")
    vision_model = torchvision.models.resnet18(weights='IMAGENET1K_V1').to(device)
    language_model = AutoModel.from_pretrained('distilbert-base-uncased').to(device)
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # 2. Load Real Data (MNIST)
    print("Loading MNIST data...")
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), # for ResNet
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mnist_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=preprocess)
    dataloader = DataLoader(mnist_dataset, batch_size=N_SAMPLES, shuffle=True)
    
    vision_data, labels = next(iter(dataloader))
    vision_data = vision_data.to(device)
    texts = [f"a digit {label.item()}" for label in labels]
    
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
    language_data_dict = {'input_ids': encoded['input_ids'].to(device), 'attention_mask': encoded['attention_mask'].to(device)}

    # 3. Run Analysis
    analyzer = AlignmentAnalyzer()
    
    print("\nRunning alignment analysis for vision model (resnet18 - layer2)...")
    vision_alignments = analyzer.compute_all_alignments(vision_model, vision_data, layer_name='layer2', model_type='vision', n_samples=N_SAMPLES)

    print("\nRunning alignment analysis for language model (distilbert - layer_2)...")
    language_alignments = analyzer.compute_all_alignments(language_model, language_data_dict, layer_name='layer_2', model_type='language', n_samples=N_SAMPLES)

    # 4. Print Results
    print("\n--- Comprehensive Alignment Analysis Results (MNIST Data) ---")
    print("\nVision Model (resnet18 - layer2):")
    print(f"  RWA (Rep-Weight) Similarity: {vision_alignments['RWA']:.4f}")
    print(f"  RGA (Rep-Gradient) Similarity: {vision_alignments['RGA']:.4f}")
    print(f"  GWA (Gradient-Weight) Similarity: {vision_alignments['GWA']:.4f}")

    print("\nLanguage Model (distilbert - layer_2):")
    print(f"  RWA (Rep-Weight) Similarity: {language_alignments['RWA']:.4f}")
    print(f"  RGA (Rep-Gradient) Similarity: {language_alignments['RGA']:.4f}")
    print(f"  GWA (Gradient-Weight) Similarity: {language_alignments['GWA']:.4f}")
