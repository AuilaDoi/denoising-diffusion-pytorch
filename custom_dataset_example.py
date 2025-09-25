"""
Example script demonstrating how to use denoising diffusion with a custom dataset
that has shape (2, 480, 480) instead of the default (3, H, W) format.

This replaces any potential MNIST usage with a custom dataset loader.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision import transforms as T
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


class CustomDataset(Dataset):
    """
    Custom dataset for (2, 480, 480) shaped data.
    This can load data from various sources and ensures output shape is (2, 480, 480).
    """
    
    def __init__(
        self,
        folder=None,
        data_array=None,
        image_size=480,
        channels=2,
        augment_horizontal_flip=False,
        normalize_to_neg_one_to_one=True
    ):
        """
        Args:
            folder: Path to folder containing data files (optional)
            data_array: Preloaded numpy array of shape (N, 2, 480, 480) (optional)
            image_size: Target image size (default: 480)
            channels: Number of channels (should be 2 for this example)
            augment_horizontal_flip: Whether to apply horizontal flip augmentation
            normalize_to_neg_one_to_one: Whether to normalize data to [-1, 1] range
        """
        super().__init__()
        
        self.image_size = image_size
        self.channels = channels
        self.normalize = normalize_to_neg_one_to_one
        
        # Load data from either folder or provided array
        if data_array is not None:
            self.data = torch.from_numpy(data_array).float()
            assert self.data.shape[1:] == (channels, image_size, image_size), \
                f"Data shape must be (N, {channels}, {image_size}, {image_size}), got {self.data.shape}"
        elif folder is not None:
            self.data = self._load_from_folder(folder)
        else:
            raise ValueError("Either 'folder' or 'data_array' must be provided")
        
        # Set up transforms
        transforms = []
        if augment_horizontal_flip:
            transforms.append(T.RandomHorizontalFlip())
        
        if self.normalize:
            transforms.append(T.Lambda(lambda x: x * 2.0 - 1.0))  # Normalize to [-1, 1]
        
        self.transform = T.Compose(transforms) if transforms else nn.Identity()
        
        print(f"CustomDataset initialized with {len(self.data)} samples of shape {self.data.shape[1:]}")
    
    def _load_from_folder(self, folder):
        """
        Load data from folder. This is a placeholder - implement based on your data format.
        Could load .npy files, .pt files, or process images to create 2-channel data.
        """
        folder = Path(folder)
        
        # Example: Look for .npy or .pt files
        npy_files = list(folder.glob("*.npy"))
        pt_files = list(folder.glob("*.pt"))
        
        if npy_files:
            # Load .npy files
            data_list = []
            for file in npy_files:
                data = np.load(file)
                if data.shape == (self.channels, self.image_size, self.image_size):
                    data_list.append(data)
                else:
                    print(f"Skipping {file}: expected shape ({self.channels}, {self.image_size}, {self.image_size}), got {data.shape}")
            
            if data_list:
                return torch.from_numpy(np.stack(data_list)).float()
        
        elif pt_files:
            # Load .pt files
            data_list = []
            for file in pt_files:
                data = torch.load(file, weights_only=True)
                if data.shape == (self.channels, self.image_size, self.image_size):
                    data_list.append(data)
                else:
                    print(f"Skipping {file}: expected shape ({self.channels}, {self.image_size}, {self.image_size}), got {data.shape}")
            
            if data_list:
                return torch.stack(data_list).float()
        
        # If no suitable files found, create some dummy data as placeholder
        print(f"No suitable data files found in {folder}. Creating synthetic data for demonstration.")
        return self._create_synthetic_data(100)  # Create 100 synthetic samples
    
    def _create_synthetic_data(self, num_samples):
        """Create synthetic data for demonstration purposes."""
        # Create synthetic 2-channel data with some structure
        data = []
        for i in range(num_samples):
            # Channel 1: Some structured pattern
            x = torch.linspace(-2, 2, self.image_size)
            y = torch.linspace(-2, 2, self.image_size)
            X, Y = torch.meshgrid(x, y, indexing='xy')
            channel1 = torch.sin(X * torch.pi) * torch.cos(Y * torch.pi) + torch.randn_like(X) * 0.1
            
            # Channel 2: Different pattern
            channel2 = torch.cos(X * torch.pi / 2) * torch.sin(Y * torch.pi / 2) + torch.randn_like(X) * 0.1
            
            # Combine channels and add random variations
            sample = torch.stack([channel1, channel2]) + torch.randn(2, self.image_size, self.image_size) * 0.05
            data.append(sample)
        
        return torch.stack(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        return self.transform(sample)


def create_custom_diffusion_model():
    """Create a diffusion model configured for (2, 480, 480) data."""
    
    # Create U-Net model with 2 channels
    model = Unet(
        dim=64,                    # Base dimension
        dim_mults=(1, 2, 4, 8),   # Dimension multipliers for each level
        channels=2,                # 2 channels for our custom data
        flash_attn=False          # Disable flash attention for compatibility
    )
    
    # Create diffusion model
    diffusion = GaussianDiffusion(
        model,
        image_size=480,            # 480x480 image size
        timesteps=1000,            # Number of diffusion steps
        sampling_timesteps=250     # Faster sampling with DDIM
    )
    
    return diffusion


def example_training_with_custom_dataset():
    """Example of training with custom dataset."""
    
    print("Creating custom dataset...")
    
    # Option 1: Create dataset from synthetic data
    synthetic_data = np.random.randn(50, 2, 480, 480).astype(np.float32)
    dataset = CustomDataset(data_array=synthetic_data)
    
    # Option 2: Create dataset from folder (commented out)
    # dataset = CustomDataset(folder="path/to/your/custom/data")
    
    print("Creating diffusion model...")
    diffusion = create_custom_diffusion_model()
    
    print("Testing dataset and model compatibility...")
    # Test with a single batch
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    
    # Test forward pass
    with torch.no_grad():
        loss = diffusion(sample.unsqueeze(0))  # Add batch dimension
    print(f"Forward pass successful! Loss: {loss.item():.4f}")
    
    # Create trainer (commented out to avoid actual training)
    """
    trainer = Trainer(
        diffusion,
        dataset=dataset,           # Pass dataset directly instead of folder
        train_batch_size=4,        # Small batch size for 480x480 images
        train_lr=1e-4,
        train_num_steps=10000,     # Reduced for demo
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=True,                  # Mixed precision to save memory
        save_and_sample_every=500
    )
    
    print("Starting training...")
    trainer.train()
    """
    
    print("Example completed successfully!")
    return diffusion, dataset


def example_sampling():
    """Example of generating samples with the custom diffusion model."""
    
    print("Creating diffusion model for sampling...")
    diffusion = create_custom_diffusion_model()
    
    print("Generating samples...")
    with torch.no_grad():
        # Generate 4 samples
        sampled_images = diffusion.sample(batch_size=4)
    
    print(f"Generated samples shape: {sampled_images.shape}")  # Should be (4, 2, 480, 480)
    
    # Save samples (optional)
    # torch.save(sampled_images, "custom_generated_samples.pt")
    
    return sampled_images


if __name__ == "__main__":
    print("=== Custom Dataset Example for (2, 480, 480) Data ===\n")
    
    # Example 1: Training setup
    print("1. Training example:")
    diffusion, dataset = example_training_with_custom_dataset()
    print()
    
    # Example 2: Sampling
    print("2. Sampling example:")
    samples = example_sampling()
    print()
    
    print("=== All examples completed successfully! ===")
    print("\nTo use with your own data:")
    print("1. Prepare your data as .npy or .pt files with shape (2, 480, 480)")
    print("2. Put them in a folder")
    print("3. Use: CustomDataset(folder='your_data_folder')")
    print("4. Or provide data directly: CustomDataset(data_array=your_numpy_array)")