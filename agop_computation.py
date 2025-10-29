import torch
import torch.func as func
from functools import partial
from typing import Callable, Optional, Union


def compute_agop_efficient(model_fn, params, X, layer_idx, 
                          get_layer_activations_fn: Optional[Callable] = None,
                          forward_from_layer_fn: Optional[Callable] = None):
    
    # Step 1: Define function to extract activations at layer
    def get_layer_activations(params, x):
        if get_layer_activations_fn is not None:
            return get_layer_activations_fn(params, x, layer_idx)
        else:
            # Try to use model_fn with layer_idx argument
            try:
                return model_fn(params, x, return_layer=layer_idx)
            except (TypeError, KeyError):
                # If that fails, assume model_fn can handle it differently
                # You may need to provide get_layer_activations_fn for your specific model
                raise ValueError(
                    "Could not extract layer activations. Please provide "
                    "get_layer_activations_fn that returns activations at layer_idx."
                )
    
    # Step 2: Define function to forward from activations to output
    def forward_from_activations(params, x, activations):
        if forward_from_layer_fn is not None:
            return forward_from_layer_fn(params, x, activations, layer_idx)
        else:
            # Try to use model_fn with start_from_layer argument
            try:
                return model_fn(params, x, start_from_layer=layer_idx, 
                              fixed_activations=activations)
            except (TypeError, KeyError):
                raise ValueError(
                    "Could not forward from layer activations. Please provide "
                    "forward_from_layer_fn that takes activations and returns output."
                )
    
    # Step 3: Compute gradient w.r.t activations for single sample
    def grad_wrt_activations_single(x):
        # Get activations first
        h = get_layer_activations(params, x)
        
        # Ensure h requires grad for gradient computation (needed for torch.func)
        if isinstance(h, torch.Tensor):
            h = h.detach().requires_grad_(True)
        
        # Define function that takes activations and returns output
        def output_given_activations(h_activations):
            output = forward_from_activations(params, x, h_activations)
            return output
        
        # Compute Jacobian: (output_dim, activation_dim)
        # Use torch.func.jacfwd for forward-mode AD (or jacrev for reverse-mode)
        # jacfwd is typically more efficient for wide outputs (k > d)
        try:
            jac = func.jacfwd(output_given_activations)(h)  # shape: (k, d)
        except RuntimeError:
            # Fallback to reverse-mode if forward-mode fails
            jac = func.jacrev(output_given_activations)(h)  # shape: (k, d)
        return jac
    
    # Step 4: Vectorize over batch using torch.func.vmap
    # torch.func.vmap takes the function and in_dims/out_dims
    batch_gradients = func.vmap(grad_wrt_activations_single, in_dims=0)(X)  # shape: (n, k, d)
    
    # Step 5: Compute outer products efficiently
    # For each sample: (∇_h f) @ (∇_h f)^T where ∇_h f has shape (k, d)
    def compute_outer_product(grad):
        # grad shape: (k, d) -> want (d, d)
        # AGOP = sum over output dimensions of grad^T @ grad
        return torch.einsum('ki,kj->ij', grad, grad)
    
    # Vectorize outer product computation
    outer_products = func.vmap(compute_outer_product, in_dims=0)(batch_gradients)  # (n, d, d)
    
    # Step 6: Average over samples
    agop = torch.mean(outer_products, dim=0)  # (d, d)
    
    return agop


def compute_agop_for_sequential_model(model_apply_fn, params, X, layer_idx):
    def get_layer_activations(params, x, layer_idx):
        raise NotImplementedError(
            "Please implement get_layer_activations based on your model structure"
        )
    
    def forward_from_layer(params, x, activations, layer_idx):
        raise NotImplementedError(
            "Please implement forward_from_layer based on your model structure"
        )
    
    return compute_agop_efficient(
        model_apply_fn, params, X, layer_idx,
        get_layer_activations_fn=get_layer_activations,
        forward_from_layer_fn=forward_from_layer
    )


def compute_agop_with_hooks(model_fn, params, X, layer_idx, 
                            extract_layer_fn: Callable,
                            forward_from_layer_fn: Callable):
    return compute_agop_efficient(
        model_fn, params, X, layer_idx,
        get_layer_activations_fn=extract_layer_fn,
        forward_from_layer_fn=forward_from_layer_fn
    )


# Alternative: More memory-efficient streaming version
def compute_agop_streaming(model_fn, params, dataloader, layer_idx,
                          get_layer_activations_fn: Optional[Callable] = None,
                          forward_from_layer_fn: Optional[Callable] = None,
                          lambda_reg=1e-4):
    agop_accumulator = None
    n_total = 0
    
    for batch_X, batch_y in dataloader:
        batch_size = batch_X.shape[0]
        
        # Compute batch contribution
        batch_agop = compute_agop_efficient(
            model_fn, params, batch_X, layer_idx,
            get_layer_activations_fn=get_layer_activations_fn,
            forward_from_layer_fn=forward_from_layer_fn
        )
        
        if agop_accumulator is None:
            agop_accumulator = batch_agop * batch_size
        else:
            agop_accumulator += batch_agop * batch_size
        
        n_total += batch_size
    
    # Return normalized AGOP
    agop = agop_accumulator / n_total
    
    # Compute W^T W according to FACT
    W_transpose_W = agop / lambda_reg
    
    return agop, W_transpose_W


# Example helper functions for common model architectures

def create_layer_extractor_for_sequential(model_layers, get_layer_params_fn=None):
    def extract_layer_fn(params, x, layer_idx):
        activations = x
        layers = model_layers if callable(model_layers) else model_layers
        
        # Handle negative indices
        if layer_idx < 0:
            layer_idx = len(layers) + layer_idx
        
        if not (0 <= layer_idx < len(layers)):
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {len(layers)})")
        
        for i, layer in enumerate(layers[:layer_idx + 1]):
            # Get parameters for this layer
            if get_layer_params_fn is not None:
                layer_params = get_layer_params_fn(params, i)
            elif isinstance(params, (list, tuple)):
                layer_params = params[i] if i < len(params) else params
            else:
                layer_params = params
            
            # Apply layer - for PyTorch models
            # activations = layer(activations)  # for nn.Module layers
            # or use functional version if layer is a callable
            activations = layer(layer_params, activations)
        
        return activations
    
    return extract_layer_fn


def create_forward_from_layer_for_sequential(model_layers, get_layer_params_fn=None):
    def forward_from_layer_fn(params, x, activations, layer_idx):
        layers = model_layers if callable(model_layers) else model_layers
        
        # Handle negative indices
        if layer_idx < 0:
            layer_idx = len(layers) + layer_idx
        
        if not (0 <= layer_idx < len(layers) - 1):
            raise ValueError(f"Cannot forward from layer_idx {layer_idx} (must be < {len(layers) - 1})")
        
        current_activations = activations
        for i, layer in enumerate(layers[layer_idx + 1:], start=layer_idx + 1):
            # Get parameters for this layer
            if get_layer_params_fn is not None:
                layer_params = get_layer_params_fn(params, i)
            elif isinstance(params, (list, tuple)):
                layer_params = params[i] if i < len(params) else params
            else:
                layer_params = params
            
            # Apply remaining layers
            current_activations = layer(layer_params, current_activations)
        
        return current_activations
    
    return forward_from_layer_fn

