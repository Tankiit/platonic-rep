import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import numpy as np
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# --- Model and Data Factories ---
def get_vision_model(name, device):
    models_map = {
        # ResNets
        'resnet18': torchvision.models.resnet18,
        'resnet34': torchvision.models.resnet34,
        'resnet50': torchvision.models.resnet50,
        'resnet101': torchvision.models.resnet101,
        'resnet152': torchvision.models.resnet152,

        # MobileNets
        'mobilenet_v2': torchvision.models.mobilenet_v2,
        'mobilenet_v3_small': torchvision.models.mobilenet_v3_small,
        'mobilenet_v3_large': torchvision.models.mobilenet_v3_large,

        # EfficientNets
        'efficientnet_b0': torchvision.models.efficientnet_b0,
        'efficientnet_b1': torchvision.models.efficientnet_b1,
        'efficientnet_b2': torchvision.models.efficientnet_b2,
        'efficientnet_b3': torchvision.models.efficientnet_b3,
        'efficientnet_b4': torchvision.models.efficientnet_b4,
        'efficientnet_b5': torchvision.models.efficientnet_b5,
        'efficientnet_b6': torchvision.models.efficientnet_b6,
        'efficientnet_b7': torchvision.models.efficientnet_b7,

        # DenseNets
        'densenet121': torchvision.models.densenet121,
        'densenet161': torchvision.models.densenet161,
        'densenet169': torchvision.models.densenet169,
        'densenet201': torchvision.models.densenet201,

        # VGG
        'vgg11': torchvision.models.vgg11,
        'vgg13': torchvision.models.vgg13,
        'vgg16': torchvision.models.vgg16,
        'vgg19': torchvision.models.vgg19,

        # Vision Transformers
        'vit_b_16': torchvision.models.vit_b_16,
        'vit_b_32': torchvision.models.vit_b_32,
        'vit_l_16': torchvision.models.vit_l_16,

        # ConvNeXt
        'convnext_tiny': torchvision.models.convnext_tiny,
        'convnext_small': torchvision.models.convnext_small,
        'convnext_base': torchvision.models.convnext_base,
        'convnext_large': torchvision.models.convnext_large,

        # Swin Transformers
        'swin_t': torchvision.models.swin_t,
        'swin_s': torchvision.models.swin_s,
        'swin_b': torchvision.models.swin_b,
    }

    if name not in models_map:
        raise ValueError(f"Unknown vision model: {name}. Available: {list(models_map.keys())}")

    return models_map[name](weights='IMAGENET1K_V1').to(device)

def get_dataset(name, preprocess, n_samples):
    datasets_config = {
        'mnist': {
            'class': torchvision.datasets.MNIST,
            'root': './data',
            'label_template': lambda label, classes: f"a digit {label}"
        },
        'fashion_mnist': {
            'class': torchvision.datasets.FashionMNIST,
            'root': './data',
            'label_template': lambda label, classes: f"a photo of {classes[label]}"
        },
        'cifar10': {
            'class': torchvision.datasets.CIFAR10,
            'root': '/Users/tanmoy/research/data',
            'label_template': lambda label, classes: f"a photo of a {classes[label]}"
        },
        'cifar100': {
            'class': torchvision.datasets.CIFAR100,
            'root': '/Users/tanmoy/research/data',
            'label_template': lambda label, classes: f"a photo of a {classes[label]}"
        },
        'svhn': {
            'class': torchvision.datasets.SVHN,
            'root': './data',
            'label_template': lambda label, classes: f"a digit {label}",
            'split': 'train'
        },
        'stl10': {
            'class': torchvision.datasets.STL10,
            'root': './data',
            'label_template': lambda label, classes: f"a photo of a {classes[label]}",
            'split': 'train'
        }
    }

    if name not in datasets_config:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(datasets_config.keys())}")

    config = datasets_config[name]

    # Load dataset with appropriate parameters
    if name == 'svhn' or name == 'stl10':
        dataset = config['class'](root=config['root'], split=config['split'], download=True, transform=preprocess)
    else:
        dataset = config['class'](root=config['root'], train=True, download=True, transform=preprocess)

    # Get class names if available
    classes = getattr(dataset, 'classes', None)

    # Generate labels
    if hasattr(dataset, 'targets'):
        targets = dataset.targets if isinstance(dataset.targets, list) else dataset.targets.tolist()
    elif hasattr(dataset, 'labels'):
        targets = dataset.labels if isinstance(dataset.labels, list) else dataset.labels.tolist()
    else:
        targets = list(range(len(dataset)))

    labels = [config['label_template'](label, classes) for label in targets]

    loader = DataLoader(dataset, batch_size=n_samples, shuffle=True)
    vision_data, label_indices = next(iter(loader))
    texts = [labels[i] for i in label_indices]
    return vision_data, texts

# --- Alignment Analyzer Class ---
class ComprehensiveAnalyzer:
    """Performs S^repr, RWA, RGA, and GWA analyses."""

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

    def compute_representation_compatibility(self, vision_model, language_model, vision_data, language_data_dict, n_samples):
        print("  Computing cross-modal S^repr...")
        vision_activations = self._get_activations(vision_model, vision_data, 'final', 'vision', n_samples)
        language_activations = self._get_activations(language_model, language_data_dict, 'final', 'language', n_samples)
        
        v_flat = vision_activations.reshape(vision_activations.shape[0], -1)
        l_flat = language_activations.reshape(language_activations.shape[0], -1)

        reg = LinearRegression().fit(v_flat, l_flat)
        return r2_score(l_flat, reg.predict(v_flat))

    def compute_intra_model_alignments(self, model, data, layer_name, model_type, n_samples, n_param_samples):
        model.eval()
        # Hc
        print(f"  Computing Representation Covariance (Hc) for {layer_name}...")
        activations = self._get_activations(model, data, layer_name, model_type, n_samples)
        hc_matrix = np.cov(activations.reshape(activations.shape[0], -1), rowvar=False)
        hc_spectrum = self._get_normalized_eigenspectrum(hc_matrix)

        # Zc
        print(f"  Computing Weight Covariance (Zc) for {layer_name}...")
        weights = self._get_weights(model, layer_name, model_type)
        zc_matrix = np.cov(weights, rowvar=True)
        zc_spectrum = self._get_normalized_eigenspectrum(zc_matrix)

        # Gc
        print(f"  Computing Approximated Gradient Covariance (Gc) for {layer_name}...")
        gradients = self._compute_model_gradients(model, data, model_type, n_samples)
        grad_flat = self._flatten_gradients(gradients)
        if grad_flat.shape[1] > n_param_samples:
            param_indices = np.random.choice(grad_flat.shape[1], n_param_samples, replace=False)
            grad_flat_sampled = grad_flat[:, param_indices]
        else:
            grad_flat_sampled = grad_flat
        gc_matrix = np.cov(grad_flat_sampled.T)
        gc_spectrum = self._get_normalized_eigenspectrum(gc_matrix)

        # Alignments
        print("  Computing alignments...")
        return {
            'RWA': self._compute_spectrum_similarity(hc_spectrum, zc_spectrum),
            'RGA': self._compute_spectrum_similarity(hc_spectrum, gc_spectrum),
            'GWA': self._compute_spectrum_similarity(gc_spectrum, zc_spectrum)
        }

    def _get_activations(self, model, data, layer_name, model_type, n_samples):
        model.eval()
        with torch.no_grad():
            if model_type == 'vision':
                x = data[:n_samples]
                if layer_name == 'final':
                    return model.features(x).mean(dim=(2,3)).cpu().numpy() if hasattr(model, 'features') else model(x).mean(dim=(2,3)).cpu().numpy()
                submodule = model.get_submodule(layer_name.rsplit('.', 1)[0])
                activations = {}
                def hook(m, i, o): activations['out'] = o
                handle = submodule.register_forward_hook(hook)
                model(x)
                handle.remove()
                return activations['out'].mean(dim=(2,3)).cpu().numpy()
            
            elif model_type == 'language':
                input_ids = data['input_ids'][:n_samples]
                attention_mask = data['attention_mask'][:n_samples]
                extended_attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0
                
                if layer_name == 'final':
                    return model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.mean(dim=1).cpu().numpy()
                
                submodule = model.get_submodule(layer_name.rsplit('.', 1)[0])
                activations = {}
                def hook(m, i, o): activations['out'] = o[0]
                handle = submodule.register_forward_hook(hook)
                model(input_ids=input_ids, attention_mask=attention_mask)
                handle.remove()
                return activations['out'].mean(dim=1).cpu().numpy()
        return None

    def _get_weights(self, model, layer_name, model_type):
        model.eval()
        submodule = model.get_submodule(layer_name)
        w = submodule.weight.detach().cpu().numpy()
        return w.reshape(w.shape[0], -1) if w.ndim > 1 else w

    def _compute_model_gradients(self, model, data, model_type, n_samples):
        model.train()
        gradients = []
        num_samples_to_process = n_samples
        if model_type == 'language':
            input_ids = data['input_ids'][:n_samples]; attention_mask = data['attention_mask'][:n_samples]
            num_samples_to_process = len(input_ids)
        else:
            data = data[:n_samples]

        for i in range(num_samples_to_process):
            model.zero_grad()
            if model_type == 'language':
                output = model(input_ids=input_ids[i:i+1], attention_mask=attention_mask[i:i+1])
            else:
                output = model(data[i:i+1])
            if isinstance(output, dict): output = output['last_hidden_state']
            loss = torch.mean(output[0])
            loss.backward()
            grad_vec = [p.grad.view(-1).detach().cpu().numpy() for p in model.parameters() if p.grad is not None]
            if grad_vec: gradients.append(np.concatenate(grad_vec))
        return np.array(gradients)

    def _flatten_gradients(self, gradients):
        return gradients.reshape(gradients.shape[0], -1)

# --- Main Execution ---
if __name__ == "__main__":

    with open('config.json', 'r') as f:
        config = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    analyzer = ComprehensiveAnalyzer()
    params = config['analysis_parameters']

    for exp in config['experiments']:
        print(f"\n--- Running Experiment: {exp['name']} ---")
        try:
            # 1. Load Models
            print("  Loading models...")
            vision_model = get_vision_model(exp['vision_model'], device)
            language_model = AutoModel.from_pretrained(exp['language_model']).to(device)
            tokenizer = AutoTokenizer.from_pretrained(exp['language_model'])

            # 2. Load Data
            print(f"  Loading {exp['dataset']} data...")
            preprocess = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            vision_data, texts = get_dataset(exp['dataset'], preprocess, params['n_samples'])
            vision_data = vision_data.to(device)
            encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
            language_data_dict = {'input_ids': encoded['input_ids'].to(device), 'attention_mask': encoded['attention_mask'].to(device)}

            # 3. Run Analyses
            s_repr = analyzer.compute_representation_compatibility(vision_model, language_model, vision_data, language_data_dict, params['n_samples'])
            
            vision_alignments = analyzer.compute_intra_model_alignments(vision_model, vision_data, exp['vision_layer'], 'vision', params['n_samples'], params['n_param_samples'])
            
            language_alignments = analyzer.compute_intra_model_alignments(language_model, language_data_dict, exp['language_layer'], 'language', params['n_samples'], params['n_param_samples'])

            # 4. Print Results
            print(f"\n--- Results for {exp['name']} ---")
            print(f"Cross-Modal Representation Compatibility (S^repr): {s_repr:.4f}")
            print(f"\nVision Model ({exp['vision_model']} - {exp['vision_layer']}):")
            print(f"  RWA (Rep-Weight) Similarity: {vision_alignments['RWA']:.4f}")
            print(f"  RGA (Rep-Gradient) Similarity: {vision_alignments['RGA']:.4f}")
            print(f"  GWA (Gradient-Weight) Similarity: {vision_alignments['GWA']:.4f}")
            print(f"\nLanguage Model ({exp['language_model']} - {exp['language_layer']}):")
            print(f"  RWA (Rep-Weight) Similarity: {language_alignments['RWA']:.4f}")
            print(f"  RGA (Rep-Gradient) Similarity: {language_alignments['RGA']:.4f}")
            print(f"  GWA (Gradient-Weight) Similarity: {language_alignments['GWA']:.4f}")

        except Exception as e:
            print(f"  !!! Experiment failed: {e}")