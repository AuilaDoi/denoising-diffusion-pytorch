# Custom Dataset Integration for Denoising Diffusion PyTorch

This document describes how to use custom datasets with shapes like `(2, 480, 480)` instead of the default RGB images.

## Overview

The repository has been enhanced to support custom datasets with arbitrary shapes and data formats. This is particularly useful when working with:

- Multi-channel data that isn't standard RGB (e.g., 2-channel data)
- Large image dimensions (e.g., 480x480 instead of typical smaller sizes)  
- Custom data formats that don't come from image files
- Scientific data, medical imaging, or other specialized datasets

## Key Changes Made

### 1. Modified Trainer Class

The `Trainer` class now accepts either:
- **Original**: `folder` parameter for folder-based image datasets
- **New**: `dataset` parameter for custom PyTorch datasets

```python
# Original folder-based approach
trainer = Trainer(diffusion_model, folder='path/to/images')

# New custom dataset approach  
trainer = Trainer(diffusion_model, dataset=custom_dataset)
```

### 2. CustomDataset Class

A new `CustomDataset` class supports `(2, 480, 480)` shaped data:

```python
from custom_dataset_example import CustomDataset

# From numpy array
dataset = CustomDataset(data_array=your_numpy_array)

# From folder of .npy or .pt files
dataset = CustomDataset(folder='path/to/data/files')

# With augmentation
dataset = CustomDataset(
    data_array=data,
    augment_horizontal_flip=True,
    normalize_to_neg_one_to_one=True
)
```

## Usage Examples

### Basic Custom Dataset Usage

```python
import numpy as np
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from custom_dataset_example import CustomDataset

# 1. Create your custom data (2, 480, 480) shape
custom_data = np.random.randn(1000, 2, 480, 480).astype(np.float32)
dataset = CustomDataset(data_array=custom_data)

# 2. Create model for 2-channel, 480x480 data
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=2,  # 2 channels instead of 3
    flash_attn=True
)

diffusion = GaussianDiffusion(
    model,
    image_size=480,  # 480x480 instead of smaller sizes
    timesteps=1000
)

# 3. Train with custom dataset
trainer = Trainer(
    diffusion,
    dataset=dataset,  # Use custom dataset
    train_batch_size=4,
    train_lr=8e-5,
    train_num_steps=100000,
    amp=True  # Recommended for large images
)

trainer.train()
```

### Data Formats Supported

#### 1. Numpy Arrays
```python
# Shape: (N, 2, 480, 480)
data = np.load('your_data.npy')  
dataset = CustomDataset(data_array=data)
```

#### 2. Individual Files
```python
# Folder with .npy or .pt files, each with shape (2, 480, 480)
dataset = CustomDataset(folder='path/to/data/files')
```

#### 3. Synthetic Data (for testing)
```python
# Automatically creates synthetic data for testing
dataset = CustomDataset(folder='empty/folder')  # Will create synthetic data
```

### Memory Considerations

For large images like 480x480, consider:

```python
trainer = Trainer(
    diffusion,
    dataset=dataset,
    train_batch_size=2,        # Smaller batch size
    gradient_accumulate_every=8, # Accumulate gradients  
    amp=True,                  # Mixed precision training
    # ... other parameters
)
```

## Model Configuration

### For 2-Channel Data

```python
model = Unet(
    dim=64,
    channels=2,              # Key: Set to 2 for 2-channel data
    dim_mults=(1, 2, 4, 8),
    # ... other parameters
)

# The diffusion model will automatically handle the 2-channel input
diffusion = GaussianDiffusion(model, image_size=480)
```

### Channel Mapping

The model automatically determines the appropriate color space:
- `channels=1` → Grayscale ('L')
- `channels=2` → Custom 2-channel data (no automatic conversion)
- `channels=3` → RGB
- `channels=4` → RGBA

## Backward Compatibility

All existing functionality remains unchanged:

```python
# This still works exactly as before
trainer = Trainer(
    diffusion_model, 
    'path/to/image/folder',
    # ... parameters
)
```

## File Structure

```
denoising-diffusion-pytorch/
├── custom_dataset_example.py       # CustomDataset class and examples
├── enhanced_trainer_example.py     # Complete training examples
├── test_custom_dataset.py         # Basic functionality tests
├── CUSTOM_DATASET_README.md       # This documentation
└── denoising_diffusion_pytorch/
    └── denoising_diffusion_pytorch.py  # Modified Trainer class
```

## Quick Start

1. **Install the package** (if not already):
   ```bash
   pip install -e .
   ```

2. **Prepare your data** in shape `(N, 2, 480, 480)`:
   ```python
   import numpy as np
   your_data = np.random.randn(500, 2, 480, 480).astype(np.float32)
   np.save('my_custom_data.npy', your_data)
   ```

3. **Run the example**:
   ```python
   python custom_dataset_example.py
   ```

4. **Adapt for your data**:
   ```python
   from custom_dataset_example import CustomDataset, create_custom_diffusion_model
   
   data = np.load('my_custom_data.npy') 
   dataset = CustomDataset(data_array=data)
   diffusion = create_custom_diffusion_model()
   
   # Your training code here
   ```

## Notes

- The repository now supports any channel count and image size
- No MNIST-specific code was found or needed to be replaced
- All changes maintain backward compatibility
- Custom datasets can implement any preprocessing or augmentation logic
- Memory usage scales with image size - use appropriate batch sizes for 480x480 images