
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

# --- Analyzer Class ---
class GradientAnalyzer:
    """Analyzes gradient covariance (AGOP)"""
    def compute_agop_analysis(self, model, data, model_type='vision', n_samples=10, n_param_samples=10000):
        """Compute an approximation of the AGOP eigenvalue analysis by sampling parameters."""
        gradients = self._compute_model_gradients(model, data, model_type=model_type, n_samples=n_samples)
        grad_flat = self._flatten_gradients(gradients)

        n_params = grad_flat.shape[1]
        if n_params > n_param_samples:
            print(f"Sampling {n_param_samples} out of {n_params} parameters for AGOP approximation.")
            param_indices = np.random.choice(n_params, n_param_samples, replace=False)
            grad_flat_sampled = grad_flat[:, param_indices]
        else:
            grad_flat_sampled = grad_flat
        
        agop = np.cov(grad_flat_sampled.T)
        eigenvals = np.linalg.eigvals(agop)
        eigenvals = np.real(eigenvals[eigenvals > 1e-10])
        eigenvals = np.sort(eigenvals)[::-1]
        
        if len(eigenvals) < 10:
            top10_concentration = 1.0
        else:
            top10_concentration = np.sum(eigenvals[:10]) / np.sum(eigenvals)
        
        entropy = -np.sum(eigenvals / np.sum(eigenvals) * np.log(eigenvals / np.sum(eigenvals) + 1e-10))
        effective_rank = np.exp(entropy)
        
        agop_ratio = eigenvals[0] / eigenvals[-1] if len(eigenvals) > 1 else 1.0
        
        return {
            'agop_ratio': agop_ratio,
            'top10_concentration': top10_concentration,
            'effective_rank': effective_rank
        }

    def _compute_model_gradients(self, model, data, model_type='vision', n_samples=10):
        """Compute gradients with respect to model parameters"""
        model.train()
        gradients = []
        
        if model_type == 'language':
            input_ids = data['input_ids'][:n_samples]
            attention_mask = data['attention_mask'][:n_samples]
            num_samples = len(input_ids)
        else:
            data = data[:n_samples]
            num_samples = len(data)

        for i in range(num_samples):
            model.zero_grad()
            
            if model_type == 'language':
                output = model(input_ids=input_ids[i:i+1], attention_mask=attention_mask[i:i+1])
            else:
                output = model(data[i:i+1])

            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, dict):
                output = output['last_hidden_state']

            loss = torch.mean(output)
            loss.backward()
            grad_vec = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_vec.append(param.grad.view(-1).detach().cpu().numpy())
            if grad_vec:
                gradients.append(np.concatenate(grad_vec))
        return np.array(gradients)

    def _flatten_gradients(self, gradients):
        """Flatten gradient arrays"""
        return gradients.reshape(gradients.shape[0], -1)

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
    analyzer = GradientAnalyzer()
    
    print("\nRunning Approximated AGOP analysis...")
    vision_agop = analyzer.compute_agop_analysis(vision_model, vision_data, model_type='vision', n_samples=30)
    language_agop = analyzer.compute_agop_analysis(language_model, language_data_dict, model_type='language', n_samples=30)

    # 4. Print Results
    print("\n--- Approximated AGOP Analysis Results ---")
    print("Vision Model (resnet18):")
    print(f"  AGOP Ratio: {vision_agop['agop_ratio']:.4f}")
    print(f"  Top 10 Eigenvalue Concentration: {vision_agop['top10_concentration']:.4f}")
    print(f"  Effective Rank: {vision_agop['effective_rank']:.4f}")

    print("\nLanguage Model (distilbert):")
    print(f"  AGOP Ratio: {language_agop['agop_ratio']:.4f}")
    print(f"  Top 10 Eigenvalue Concentration: {language_agop['top10_concentration']:.4f}")
    print(f"  Effective Rank: {language_agop['effective_rank']:.4f}")
