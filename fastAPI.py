from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
from pathlib import Path
import yaml
import os

# Import your CLCC model
from model.base import CLCC

app = FastAPI(title="AquaEnhance API", version="1.0.0")

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = {
    'channel_scale': 64,
    'main_ks': 3,
    'gcc_ks': 3,
    'image_size': 256
}

def find_model_file():
    """
    Search for model files in common locations
    Returns the path if found, None otherwise
    """
    possible_paths = [
        r"C:\Users\rutik\Aqua-Enhance\log\AquaEnhance\base\ckpt\best_ssim.pth",
        "checkpoints/best_model.pth",
        "checkpoints/best_psnr.pth",
        "log/AquaEnhance/base/ckpt/best_ssim.pth",
        "models/best_psnr.pth",
        "best_psnr.pth",
        "model_best.pth",
    ]
    
    print("\n🔍 Searching for model file...")
    for path in possible_paths:
        if Path(path).exists():
            print(f"✓ Found model at: {path}")
            return path
        else:
            print(f"✗ Not found: {path}")
    
    # Search in current directory and subdirectories
    print("\n🔍 Searching in current directory tree...")
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pth') or file.endswith('.pt'):
                full_path = os.path.join(root, file)
                print(f"✓ Found .pth file: {full_path}")
                return full_path
    
    return None

def load_model(model_path):
    """
    Load the trained CLCC model with comprehensive error handling
    """
    try:
        print(f"\n📦 Loading model from: {model_path}")
        
        # Initialize model with your architecture
        model = CLCC(
            channel_scale=CONFIG['channel_scale'],
            main_ks=CONFIG['main_ks'],
            gcc_ks=CONFIG['gcc_ks']
        )
        
        # Load the trained weights
        print("📥 Loading checkpoint...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            print(f"📋 Checkpoint keys: {checkpoint.keys()}")
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("✓ Loaded from 'model_state_dict'")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print("✓ Loaded from 'state_dict'")
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
                print("✓ Loaded from 'model'")
            else:
                # Try loading the dict directly
                model.load_state_dict(checkpoint)
                print("✓ Loaded checkpoint directly")
        else:
            model.load_state_dict(checkpoint)
            print("✓ Loaded checkpoint directly")
        
        model.to(device)
        model.eval()
        
        print(f"✓ Model loaded successfully on {device}")
        print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        raise
    except KeyError as e:
        print(f"❌ Key error in checkpoint: {e}")
        print("   The checkpoint structure doesn't match expected format")
        raise
    except RuntimeError as e:
        print(f"❌ Runtime error loading model: {e}")
        print("   This usually means architecture mismatch between saved model and code")
        raise
    except Exception as e:
        print(f"❌ Unexpected error loading model: {type(e).__name__}: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model
    
    print("\n" + "="*60)
    print("🌊 AquaEnhance API Starting...")
    print("="*60)
    print(f"Device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*60)
    
    try:
        # Try to find the model file
        model_path = find_model_file()
        
        if model_path is None:
            print("\n❌ ERROR: No model file found!")
            print("\n📋 Instructions to fix:")
            print("1. Locate your trained model file (.pth or .pt)")
            print("2. Place it in one of these locations:")
            print("   - checkpoints/best_psnr.pth")
            print("   - models/best_psnr.pth")
            print("   - log/AquaEnhance/base/ckpt/best_psnr.pth")
            print("3. Or update the model_path in the code")
            print("\n⚠️  API will run but /enhance endpoint won't work until model is loaded")
            print("="*60 + "\n")
            return
        
        model = load_model(model_path)
        
        print("\n" + "="*60)
        print("✅ AquaEnhance model ready!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during startup: {type(e).__name__}")
        print(f"   {str(e)}")
        print("\n⚠️  The API will still run but enhancement won't work until model is loaded")
        print("="*60 + "\n")

def preprocess_image(image: Image.Image):
    """Preprocess image for CLCC model input"""
    original_size = image.size
    
    transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    return img_tensor.to(device), original_size

def postprocess_output(output_tensor, original_size):
    """Convert model output back to image"""
    output = output_tensor.squeeze(0).cpu().detach()
    
    # CLCC uses Tanh activation, so output is in [-1, 1]
    output = (output + 1) / 2.0
    output = torch.clamp(output, 0, 1)
    
    output_np = output.numpy().transpose(1, 2, 0)
    output_np = (output_np * 255).astype(np.uint8)
    
    enhanced_image = Image.fromarray(output_np)
    enhanced_image = enhanced_image.resize(original_size, Image.LANCZOS)
    
    return enhanced_image

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "AquaEnhance API is operational",
        "model_loaded": model is not None,
        "device": str(device),
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_architecture": "CLCC",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "config": CONFIG,
        "model_loaded": model is not None
    }

@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    """Enhance underwater image using CLCC model"""
    
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs for details. The model file may be missing or there was an error loading it."
        )
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type: {file.content_type}. Please upload an image."
        )
    
    try:
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        print(f"📸 Processing: {file.filename} | Size: {input_image.size}")
        
        input_tensor, original_size = preprocess_image(input_image)
        
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        enhanced_image = postprocess_output(output_tensor, original_size)
        
        img_byte_arr = io.BytesIO()
        enhanced_image.save(img_byte_arr, format='PNG', quality=95)
        img_byte_arr.seek(0)
        
        print(f"✓ Enhancement complete for {file.filename}")
        
        return StreamingResponse(
            img_byte_arr, 
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=enhanced_{file.filename}",
                "X-Processing-Status": "success"
            }
        )
        
    except Exception as e:
        print(f"❌ Error processing {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing image: {str(e)}"
        )

@app.post("/compare")
async def compare_images(file: UploadFile = File(...)):
    """Return both original and enhanced images for comparison"""
    
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        import base64
        
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        input_tensor, original_size = preprocess_image(input_image)
        
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        enhanced_image = postprocess_output(output_tensor, original_size)
        
        def image_to_base64(img):
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "original": image_to_base64(input_image),
            "enhanced": image_to_base64(enhanced_image),
            "filename": file.filename,
            "original_size": original_size
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing image: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🌊 Starting AquaEnhance FastAPI Server")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")