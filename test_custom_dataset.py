"""
Quick test of the custom dataset implementation
"""

import torch
import numpy as np
from custom_dataset_example import CustomDataset, create_custom_diffusion_model

def test_custom_dataset():
    print("Testing CustomDataset...")
    
    # Create synthetic data
    synthetic_data = np.random.randn(10, 2, 480, 480).astype(np.float32)
    dataset = CustomDataset(data_array=synthetic_data)
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Sample shape: {dataset[0].shape}")
    
    # Test model compatibility
    diffusion = create_custom_diffusion_model()
    print(f"Model channels: {diffusion.model.channels}")
    print(f"Model image size: {diffusion.image_size}")
    
    # Test forward pass
    sample = dataset[0]
    with torch.no_grad():
        loss = diffusion(sample.unsqueeze(0))
    print(f"Forward pass successful! Loss: {loss.item():.4f}")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_custom_dataset()