#!/usr/bin/env python3
"""
Simple test to verify Dask integration works correctly.
This script simulates the image loading part of the inference pipeline
without requiring the full model or PyTorch.
"""

import tempfile
from pathlib import Path
import numpy as np
import dask
import dask.bag as db
from dask.diagnostics import ProgressBar


def create_dummy_images(output_dir, num_images=5):
    """Create dummy image files for testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = []
    for i in range(num_images):
        # Create a simple numpy array to simulate an image
        dummy_data = np.random.rand(10, 10, 10)
        image_path = output_dir / f"test_image_{i}.npy"
        np.save(image_path, dummy_data)
        image_paths.append(image_path)
    
    return image_paths


def load_and_preprocess(args):
    """Simulate image loading and preprocessing."""
    idx, image_path = args
    
    # Load image (simulated)
    image_data = np.load(image_path)
    
    # Simulate some preprocessing
    processed = image_data * 2.0
    
    return {
        'idx': idx,
        'image_path': image_path,
        'shape': image_data.shape,
        'mean': float(np.mean(processed))
    }


def test_dask_parallel_loading():
    """Test Dask parallel image loading."""
    print("Testing Dask parallel image loading...")
    
    # Create temporary directory with dummy images
    with tempfile.TemporaryDirectory() as temp_dir:
        image_paths = create_dummy_images(temp_dir, num_images=10)
        print(f"Created {len(image_paths)} dummy images")
        
        # Configure Dask
        dask.config.set(scheduler='threads')
        
        # Create Dask bag for parallel loading
        image_args = list(enumerate(image_paths))
        bag = db.from_sequence(image_args, npartitions=min(len(image_paths), 4))
        
        # Load images in parallel
        print("Loading images in parallel with Dask...")
        with ProgressBar():
            loaded_data = bag.map(load_and_preprocess).compute()
        
        # Verify results
        print(f"\nSuccessfully loaded {len(loaded_data)} images")
        print("Sample results:")
        for i, data in enumerate(loaded_data[:3]):
            print(f"  Image {data['idx']}: shape={data['shape']}, mean={data['mean']:.4f}")
        
        print("\n✓ Dask integration test passed!")
        return True


if __name__ == "__main__":
    test_dask_parallel_loading()
