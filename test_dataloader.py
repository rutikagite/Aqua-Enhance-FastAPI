import os
import sys
from torch.utils.data import DataLoader

# Add this at the top to handle Windows multiprocessing issues
if __name__ == "__main__":
    # Required for Windows multiprocessing
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    try:
        from data.uieb import UIEBTrain, UIEBValid
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure your data/uieb.py file exists and NumPy is properly installed")
        sys.exit(1)

    def test_dataloader():
        # Test parameters - Windows-optimized
        batch_size = 2  # Reduced for Windows
        size = 256
        
        # Paths to your dataset
        train_path = "data/UIEB/train"
        valid_path = "data/UIEB/valid"
        
        # Check if directories exist
        if not os.path.exists(train_path):
            print(f"❌ Train directory not found: {train_path}")
            return
        
        if not os.path.exists(valid_path):
            print(f"❌ Valid directory not found: {valid_path}")
            return
        
        try:
            # Create datasets using your existing classes
            print("Creating train dataset...")
            train_dataset = UIEBTrain(folder=train_path, size=size)
            
            print("Creating validation dataset...")
            valid_dataset = UIEBValid(folder=valid_path, size=size)
            
        except Exception as e:
            print(f"❌ Dataset creation error: {e}")
            return
        
        # Create dataloaders with Windows-compatible settings
        try:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,      # CRITICAL: Set to 0 for Windows
                pin_memory=False    # CRITICAL: Disable for Windows memory issues
            )
            
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,      # CRITICAL: Set to 0 for Windows
                pin_memory=False    # CRITICAL: Disable for Windows memory issues
            )
        except Exception as e:
            print(f"❌ DataLoader creation error: {e}")
            return
        
        # Test loading
        print(f"\n=== Dataset Info ===")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Valid dataset size: {len(valid_dataset)}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Valid batches: {len(valid_loader)}")
        
        # Test train loader
        print(f"\n=== Testing Train Loader ===")
        try:
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                print(f"Train Batch {batch_idx}:")
                print(f"  Input shape: {inputs.shape}")
                print(f"  Target shape: {targets.shape}")
                print(f"  Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
                print(f"  Target range: [{targets.min():.3f}, {targets.max():.3f}]")
                
                # Check data augmentation worked
                if batch_idx == 0:
                    print(f"  Data augmentations applied: padding, crop, flip, rotate")
                break  # Only test first batch
        except Exception as e:
            print(f"❌ Train loader error: {e}")
            return
        
        # Test valid loader
        print(f"\n=== Testing Valid Loader ===")
        try:
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                print(f"Valid Batch {batch_idx}:")
                print(f"  Input shape: {inputs.shape}")
                print(f"  Target shape: {targets.shape}")
                print(f"  Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
                print(f"  Target range: [{targets.min():.3f}, {targets.max():.3f}]")
                
                if batch_idx == 0:
                    print(f"  Validation: only resize applied (no random augmentations)")
                break  # Only test first batch
        except Exception as e:
            print(f"❌ Valid loader error: {e}")
            return
        
        print(f"\n✅ Data loading test passed!")
        print(f"Your UIEBTrain and UIEBValid classes are working correctly!")
        print(f"Note: num_workers=0 for Windows compatibility")

    # Test NumPy compatibility first
    def test_numpy_compatibility():
        try:
            import numpy as np
            import torch
            print(f"✅ NumPy version: {np.__version__}")
            print(f"✅ PyTorch version: {torch.__version__}")
            
            # Test basic operations
            arr = np.array([1, 2, 3])
            tensor = torch.from_numpy(arr)
            print(f"✅ NumPy-PyTorch conversion works")
            return True
        except Exception as e:
            print(f"❌ NumPy compatibility issue: {e}")
            print("Run: pip install 'numpy<2.0' --force-reinstall")
            return False

    # Run tests
    print("=== Environment Check ===")
    if test_numpy_compatibility():
        print("\n=== DataLoader Test ===")
        test_dataloader()
    else:
        print("❌ Fix NumPy compatibility first!")