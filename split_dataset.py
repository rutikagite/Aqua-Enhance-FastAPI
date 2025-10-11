import os
import shutil
import random
from pathlib import Path

def split_uieb_dataset(
    raw_dir="data/UIEB/raw/raw-890",
    reference_dir="data/UIEB/reference/reference-890", 
    output_dir="data/UIEB",
    train_ratio=0.7,
    valid_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    """
    Split UIEB dataset into train/valid/test sets
    
    Args:
        raw_dir: Directory containing degraded images
        reference_dir: Directory containing enhanced reference images
        output_dir: Output directory for organized dataset
        train_ratio: Proportion for training (0.7 = 70%)
        valid_ratio: Proportion for validation (0.15 = 15%) 
        test_ratio: Proportion for testing (0.15 = 15%)
        seed: Random seed for reproducible splits
    """
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Get all image files
    raw_images = sorted([f for f in os.listdir(raw_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    ref_images = sorted([f for f in os.listdir(reference_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    # Ensure we have matching pairs
    assert len(raw_images) == len(ref_images), f"Mismatch: {len(raw_images)} raw vs {len(ref_images)} reference images"
    
    # Create paired list and shuffle
    image_pairs = list(zip(raw_images, ref_images))
    random.shuffle(image_pairs)
    
    # Calculate split indices
    total_images = len(image_pairs)
    train_end = int(total_images * train_ratio)
    valid_end = train_end + int(total_images * valid_ratio)
    
    # Split the data
    train_pairs = image_pairs[:train_end]
    valid_pairs = image_pairs[train_end:valid_end]
    test_pairs = image_pairs[valid_end:]
    
    print(f"Dataset split:")
    print(f"  Total images: {total_images}")
    print(f"  Training: {len(train_pairs)} ({len(train_pairs)/total_images:.1%})")
    print(f"  Validation: {len(valid_pairs)} ({len(valid_pairs)/total_images:.1%})")
    print(f"  Testing: {len(test_pairs)} ({len(test_pairs)/total_images:.1%})")
    
    # Create directory structure
    splits = ['train', 'valid', 'test']
    for split in splits:
        os.makedirs(f"{output_dir}/{split}/input", exist_ok=True)
        os.makedirs(f"{output_dir}/{split}/target", exist_ok=True)
    
    # Copy files to appropriate directories
    def copy_split(pairs, split_name):
        for i, (raw_img, ref_img) in enumerate(pairs):
            # Generate new filenames with split prefix
            new_name = f"{split_name}_{i+1:03d}.jpg"
            
            # Copy raw image to input directory
            shutil.copy2(
                os.path.join(raw_dir, raw_img),
                os.path.join(output_dir, split_name, "input", new_name)
            )
            
            # Copy reference image to target directory  
            shutil.copy2(
                os.path.join(reference_dir, ref_img),
                os.path.join(output_dir, split_name, "target", new_name)
            )
    
    # Perform the copying
    copy_split(train_pairs, 'train')
    copy_split(valid_pairs, 'valid')  
    copy_split(test_pairs, 'test')
    
    print("Dataset split completed successfully!")
    
    # Verify the split
    for split in splits:
        input_count = len(os.listdir(f"{output_dir}/{split}/input"))
        target_count = len(os.listdir(f"{output_dir}/{split}/target"))
        print(f"  {split}: {input_count} input, {target_count} target images")

if __name__ == "__main__":
    split_uieb_dataset()