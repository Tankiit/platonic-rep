
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from transformers import AutoModel, AutoTokenizer
import numpy as np
from PIL import Image
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# --- Data Creation ---
def create_test_data(n_samples: int = 50):
    """Create simple test data"""
    images = []
    texts = []
    for i in range(n_samples):
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        color_type = i % 4
        if color_type == 0:
            img_array[:, :, 0] = 255
            text = "A red image"
        elif color_type == 1:
            img_array[:, :, 1] = 255
            text = "A green image"
        elif color_type == 2:
            img_array[:, :, 2] = 255
            text = "A blue image"
        else:
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            text = "A colorful random image"
        images.append(Image.fromarray(img_array))
        texts.append(text)
    return images, texts

# --- RWA Analyzer Class ---
class RWAAnalyzer:
    """Representation-Weight Alignment (RWA) Analysis"""

    def compute_rwa_analysis(self, model, data, layer_name, model_type='vision', n_samples=30):
        """Computes the RWA for a given layer."""
        model.eval()

        # 1. Get Activations and compute Hc
        activations = self._get_activations(model, data, layer_name, model_type, n_samples)
        activations_flat = activations.reshape(activations.shape[0], -1)
        hc_matrix = np.cov(activations_flat, rowvar=False)
        hc_spectrum = self._get_normalized_eigenspectrum(hc_matrix)

        # 2. Get Weights and compute Zc
        weights = self._get_weights(model, layer_name, model_type)
        zc_matrix = np.cov(weights, rowvar=True)
        zc_spectrum = self._get_normalized_eigenspectrum(zc_matrix)

        # 3. Compare spectra
        similarity = self._compute_spectrum_similarity(hc_spectrum, zc_spectrum)
        
        return {
            'rwa_similarity': similarity,
            'hc_spectrum_shape': hc_spectrum.shape,
            'zc_spectrum_shape': zc_spectrum.shape
        }

    def _get_activations(self, model, data, layer_name, model_type, n_samples):
        model.eval()
        with torch.no_grad():
            if model_type == 'vision':
                x = data[:n_samples]
                if layer_name == 'layer2':
                    x = model.conv1(x)
                    x = model.bn1(x)
                    x = model.relu(x)
                    x = model.maxpool(x)
                    x = model.layer1(x)
                    x = model.layer2(x)
                    return x.mean(dim=(2,3)).cpu().numpy()
                else:
                    raise NotImplementedError(f"Activation extraction for layer {layer_name} not implemented for resnet18")
            
            elif model_type == 'language':
                input_ids = data['input_ids'][:n_samples]
                attention_mask = data['attention_mask'][:n_samples].to(torch.bool)
                if layer_name == 'layer_2':
                    hidden_states = model.embeddings(input_ids)
                    for i in range(3): # Up to layer_2 (0, 1, 2)
                        hidden_states = model.transformer.layer[i](hidden_states, attention_mask)[0]
                    return hidden_states.mean(dim=1).cpu().numpy()
                else:
                    raise NotImplementedError(f"Activation extraction for layer {layer_name} not implemented for distilbert")
        return None

    def _get_weights(self, model, layer_name, model_type):
        model.eval()
        if model_type == 'vision':
            if layer_name == 'layer2':
                w = model.layer2[0].conv1.weight.detach().cpu().numpy()
                return w.reshape(w.shape[0], -1) # (out_channels, in_channels * K * K)
            else:
                raise NotImplementedError(f"Weight extraction for layer {layer_name} not implemented for resnet18")
        
        elif model_type == 'language':
            if layer_name == 'layer_2':
                w = model.transformer.layer[2].attention.out_lin.weight.detach().cpu().numpy()
                return w # Already 2D
            else:
                raise NotImplementedError(f"Weight extraction for layer {layer_name} not implemented for distilbert")
        return None

    def _get_normalized_eigenspectrum(self, matrix):
        eigenvals = np.linalg.eigvals(matrix)
        eigenvals = np.real(eigenvals[eigenvals > 1e-10])
        eigenvals = np.sort(eigenvals)[::-1]
        return eigenvals / np.sum(eigenvals)

    def _compute_spectrum_similarity(self, spec1, spec2):
        len1, len2 = len(spec1), len(spec2)
        if len1 > len2:
            spec2 = np.pad(spec2, (0, len1 - len2), 'constant')
        elif len2 > len1:
            spec1 = np.pad(spec1, (0, len2 - len1), 'constant')
        
        return np.dot(spec1, spec2) / (np.linalg.norm(spec1) * np.linalg.norm(spec2))

# --- Main Execution ---
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load Models
    print("Loading models...")
    vision_model = torchvision.models.resnet18(weights='IMAGENET1K_V1').to(device)
    language_model = AutoModel.from_pretrained('distilbert-base-uncased').to(device)
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # 2. Create and Preprocess Data
    print("Creating and preprocessing data...")
    images, texts = create_test_data(n_samples=30)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    vision_data = torch.stack([preprocess(img) for img in images]).to(device)
    
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
    language_data_dict = {
        'input_ids': encoded['input_ids'].to(device),
        'attention_mask': encoded['attention_mask'].to(device)
    }

    # 3. Run Analysis
    analyzer = RWAAnalyzer()
    
    print("\nRunning RWA analysis for vision model...")
    vision_rwa = analyzer.compute_rwa_analysis(
        vision_model, vision_data, layer_name='layer2', model_type='vision', n_samples=30)

    print("\nRunning RWA analysis for language model...")
    language_rwa = analyzer.compute_rwa_analysis(
        language_model, language_data_dict, layer_name='layer_2', model_type='language', n_samples=30)

    # 4. Print Results
    print("\n--- Representation-Weight Alignment (RWA) Analysis Results ---")
    print("Vision Model (resnet18 - layer2):")
    print(f"  RWA Similarity: {vision_rwa['rwa_similarity']:.4f}")
    print(f"  Representation Spectrum Shape (Hc): {vision_rwa['hc_spectrum_shape']}")
    print(f"  Weight Spectrum Shape (Zc): {vision_rwa['zc_spectrum_shape']}")

    print("\nLanguage Model (distilbert - layer_2):")
    print(f"  RWA Similarity: {language_rwa['rwa_similarity']:.4f}")
    print(f"  Representation Spectrum Shape (Hc): {language_rwa['hc_spectrum_shape']}")
    print(f"  Weight Spectrum Shape (Zc): {language_rwa['zc_spectrum_shape']}")
