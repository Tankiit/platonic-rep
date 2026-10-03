import numpy as np
import os

base_dir = "/Users/tanmoy/research/Perceptual_Features/platonic-rep/complete_features"
models = ["efficientnet_b0", "mobilenet_v2", "resnet18", "squeezenet", "albert", "distilbert"]

print("Inspecting embedding shapes...")
for model in models:
    model_path = os.path.join(base_dir, model)
    if os.path.isdir(model_path):
        print(f"\nModel: {model}")
        for layer_file in sorted(os.listdir(model_path)):
            if layer_file.endswith(".npy"):
                try:
                    layer_path = os.path.join(model_path, layer_file)
                    embeddings = np.load(layer_path)
                    if embeddings.ndim > 2:
                        embeddings = embeddings.reshape(embeddings.shape[0], -1)
                    print(f"  - {layer_file}: {embeddings.shape}")
                except Exception as e:
                    print(f"  - Could not load {layer_file}: {e}")
