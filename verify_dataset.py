import os
from PIL import Image

def verify_dataset(data_dir="data/UIEB"):
    """Verify dataset integrity and structure"""
    
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        input_dir = f"{data_dir}/{split}/input"
        target_dir = f"{data_dir}/{split}/target"
        
        if not os.path.exists(input_dir) or not os.path.exists(target_dir):
            print(f"Missing directories for {split}")
            continue
            
        input_files = sorted(os.listdir(input_dir))
        target_files = sorted(os.listdir(target_dir))
        
        print(f"\n=== {split.upper()} SET ===")
        print(f"Input images: {len(input_files)}")
        print(f"Target images: {len(target_files)}")
        
        if len(input_files) != len(target_files):
            print(f"Mismatch in {split}: {len(input_files)} vs {len(target_files)}")
            continue
            
        # Check first few images
        for i in range(min(3, len(input_files))):
            input_path = os.path.join(input_dir, input_files[i])
            target_path = os.path.join(target_dir, target_files[i])
            
            try:
                # Check if images can be opened
                with Image.open(input_path) as img:
                    input_size = img.size
                with Image.open(target_path) as img:
                    target_size = img.size
                    
                print(f"  {input_files[i]}: {input_size} -> {target_size}")
                
            except Exception as e:
                print(f"Error with {input_files[i]}: {e}")

if __name__ == "__main__":
    verify_dataset()