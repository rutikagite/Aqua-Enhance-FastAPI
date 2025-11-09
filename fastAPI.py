from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter
import io
import numpy as np
import cv2
from pathlib import Path
import os

# Import your CLCC model
from model.base import CLCC

app = FastAPI(title="AquaEnhance API", version="1.0.0")

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
    'image_size': 256,
    # Moderate parameters for natural look
    'color_enhance': 1.4,      # Reduced from 1.6
    'contrast_enhance': 1.2,   # Reduced from 1.4
    'sharpness_enhance': 1.4,  # Reduced from 1.8
    'processing_mode': 'balanced'  # Options: 'ultra', 'balanced', 'simple', 'none'
}

def find_model_file():
    """Search for model files - prioritize SSIM checkpoint"""
    possible_paths = [
        # SSIM checkpoint (best for sharpness)
        r"C:\Users\rutik\Aqua-Enhance-FastAPI\log\AquaEnhance\base\ckpt\best_ssim.pth",
        "log/AquaEnhance/base/ckpt/best_ssim.pth",
        "checkpoints/best_ssim.pth",
        # PSNR checkpoint (backup)
        r"C:\Users\rutik\Aqua-Enhance-FastAPI\log\AquaEnhance\base\ckpt\best_psnr.pth",
        "checkpoints/best_psnr.pth",
        "log/AquaEnhance/base/ckpt/best_psnr.pth",
        # Other checkpoints
        "checkpoints/best_model.pth",
        "models/best_psnr.pth",
        "best_psnr.pth",
        "model_best.pth",
    ]
    
    print("\n🔍 Searching for model file (prioritizing SSIM checkpoint)...")
    for path in possible_paths:
        if Path(path).exists():
            print(f"✓ Found model at: {path}")
            return path
    
    print("\n🔍 Searching in current directory tree...")
    for root, dirs, files in os.walk('.'):
        for file in files:
            if 'ssim' in file.lower() and (file.endswith('.pth') or file.endswith('.pt')):
                full_path = os.path.join(root, file)
                print(f"✓ Found SSIM checkpoint: {full_path}")
                return full_path
            elif file.endswith('.pth') or file.endswith('.pt'):
                full_path = os.path.join(root, file)
                print(f"✓ Found .pth file: {full_path}")
                return full_path
    
    return None

def load_model(model_path):
    """Load the trained CLCC model"""
    try:
        print(f"\n📦 Loading model from: {model_path}")
        
        model = CLCC(
            channel_scale=CONFIG['channel_scale'],
            main_ks=CONFIG['main_ks'],
            gcc_ks=CONFIG['gcc_ks']
        )
        
        print("📥 Loading checkpoint...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
            
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
                model.load_state_dict(checkpoint)
                print("✓ Loaded checkpoint directly")
        else:
            model.load_state_dict(checkpoint)
            print("✓ Loaded checkpoint directly")
        
        model.to(device)
        model.eval()
        
        # CRITICAL: Freeze BatchNorm layers properly
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.InstanceNorm2d):
                module.eval()
                if hasattr(module, 'track_running_stats'):
                    module.track_running_stats = False
        
        # Disable gradient computation
        torch.set_grad_enabled(False)
        
        print(f"✓ Model loaded successfully on {device}")
        print(f"✓ Model in eval mode: {not model.training}")
        print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    except Exception as e:
        print(f"❌ Error loading model: {type(e).__name__}: {e}")
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
    print(f"Enhancement Settings:")
    print(f"  - Color Boost: {CONFIG['color_enhance']}x")
    print(f"  - Contrast Boost: {CONFIG['contrast_enhance']}x")
    print(f"  - Sharpness Boost: {CONFIG['sharpness_enhance']}x")
    print(f"  - Processing Mode: {CONFIG['processing_mode']}")
    print("="*60)
    
    try:
        model_path = find_model_file()
        
        if model_path is None:
            print("\n❌ ERROR: No model file found!")
            print("\n📋 Please ensure your model checkpoint is in the correct location")
            print("="*60 + "\n")
            return
        
        model = load_model(model_path)
        
        print("\n" + "="*60)
        print("✅ AquaEnhance model ready!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during startup: {type(e).__name__}")
        print(f"   {str(e)}")
        print("="*60 + "\n")

def preprocess_image(image: Image.Image):
    """
    FIXED: Preprocess image to match training pipeline
    Model expects input in [0, 1] range (converted to tensor)
    """
    original_size = image.size
    
    transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size']), 
                         interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),  # Converts to [0, 1]
        # DO NOT normalize - model wasn't trained with normalization
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    return img_tensor.to(device), original_size

def ultra_sharp_postprocess(output_tensor, original_size):
    """
    Balanced post-processing that preserves natural look while adding sharpness
    """
    # Properly convert from Tanh output [-1, 1] to [0, 1]
    output = output_tensor.squeeze(0).cpu().detach()
    output = torch.clamp(output, -1, 1)
    output = (output + 1.0) / 2.0
    output = torch.clamp(output, 0, 1)
    
    output_np = output.numpy().transpose(1, 2, 0)
    output_np = (output_np * 255).astype(np.uint8)
    
    # Convert to PIL first
    enhanced_image = Image.fromarray(output_np)
    
    # Resize with high-quality interpolation
    enhanced_image = enhanced_image.resize(original_size, Image.LANCZOS)
    
    # Convert to OpenCV for advanced processing
    enhanced_array = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)
    
    # --- BALANCED POST-PROCESSING ---
    
    # 1. Gentle unsharp mask (single pass with moderate strength)
    gaussian_blur = cv2.GaussianBlur(enhanced_array, (0, 0), 1.5)
    enhanced_array = cv2.addWeighted(enhanced_array, 1.5, gaussian_blur, -0.5, 0)
    enhanced_array = np.clip(enhanced_array, 0, 255).astype(np.uint8)
    
    # 2. Moderate CLAHE for local contrast
    lab = cv2.cvtColor(enhanced_array, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    lab = cv2.merge([l, a, b])
    enhanced_array = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 3. Very light bilateral filter to reduce noise while keeping edges
    enhanced_array = cv2.bilateralFilter(enhanced_array, 5, 50, 50)
    
    # Convert back to PIL
    enhanced_image = Image.fromarray(cv2.cvtColor(enhanced_array, cv2.COLOR_BGR2RGB))
    
    # 4. PIL enhancements (moderate values)
    enhancer_color = ImageEnhance.Color(enhanced_image)
    enhanced_image = enhancer_color.enhance(CONFIG['color_enhance'])
    
    enhancer_contrast = ImageEnhance.Contrast(enhanced_image)
    enhanced_image = enhancer_contrast.enhance(CONFIG['contrast_enhance'])
    
    enhancer_sharp = ImageEnhance.Sharpness(enhanced_image)
    enhanced_image = enhancer_sharp.enhance(CONFIG['sharpness_enhance'])
    
    # 5. Final gentle unsharp mask
    enhanced_image = enhanced_image.filter(
        ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3)
    )
    
    return enhanced_image

def simple_postprocess(output_tensor, original_size):
    """
    Conservative processing - minimal enhancement, natural look
    Use this if you're getting artifacts
    """
    output = output_tensor.squeeze(0).cpu().detach()
    
    # Properly convert Tanh output
    output = torch.clamp(output, -1, 1)
    output = (output + 1.0) / 2.0
    output = torch.clamp(output, 0, 1)
    
    output_np = output.numpy().transpose(1, 2, 0)
    output_np = (output_np * 255).astype(np.uint8)
    
    enhanced_image = Image.fromarray(output_np)
    enhanced_image = enhanced_image.resize(original_size, Image.LANCZOS)
    
    # Very gentle enhancements only
    enhancer_color = ImageEnhance.Color(enhanced_image)
    enhanced_image = enhancer_color.enhance(1.2)
    
    enhancer_contrast = ImageEnhance.Contrast(enhanced_image)
    enhanced_image = enhancer_contrast.enhance(1.15)
    
    enhancer_sharp = ImageEnhance.Sharpness(enhanced_image)
    enhanced_image = enhancer_sharp.enhance(1.3)
    
    return enhanced_image

def no_postprocess(output_tensor, original_size):
    """
    FIXED: Raw output with just proper range conversion
    Use this to diagnose if blur is from model or post-processing
    """
    output = output_tensor.squeeze(0).cpu().detach()
    
    # Properly convert Tanh output
    output = torch.clamp(output, -1, 1)
    output = (output + 1.0) / 2.0
    output = torch.clamp(output, 0, 1)
    
    output_np = output.numpy().transpose(1, 2, 0)
    output_np = (output_np * 255).astype(np.uint8)
    
    enhanced_image = Image.fromarray(output_np)
    enhanced_image = enhanced_image.resize(original_size, Image.LANCZOS)
    
    return enhanced_image

def postprocess_output(output_tensor, original_size):
    """Route to appropriate post-processing"""
    mode = CONFIG.get('processing_mode', 'balanced')
    
    if mode == 'ultra':
        return ultra_sharp_postprocess(output_tensor, original_size)
    elif mode == 'balanced':
        return ultra_sharp_postprocess(output_tensor, original_size)  # Now uses balanced version
    elif mode == 'simple':
        return simple_postprocess(output_tensor, original_size)
    elif mode == 'none':
        return no_postprocess(output_tensor, original_size)
    else:
        return ultra_sharp_postprocess(output_tensor, original_size)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "AquaEnhance API is operational",
        "model_loaded": model is not None,
        "device": str(device),
        "version": "2.0.0",
        "enhancement_config": CONFIG
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    model_info = {}
    if model is not None:
        model_info = {
            "training_mode": model.training,
            "parameters": sum(p.numel() for p in model.parameters()),
        }
    
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_architecture": "CLCC",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "config": CONFIG,
        "model_info": model_info
    }

@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    """Enhance underwater image"""
    
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Please upload an image."
        )
    
    try:
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        print(f"📸 Processing: {file.filename} | Size: {input_image.size}")
        
        # Preprocess
        input_tensor, original_size = preprocess_image(input_image)
        
        # Model inference
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        print(f"   Model output range: [{output_tensor.min():.3f}, {output_tensor.max():.3f}]")
        
        # Post-process
        enhanced_image = postprocess_output(output_tensor, original_size)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        enhanced_image.save(img_byte_arr, format='PNG', quality=95)
        img_byte_arr.seek(0)
        
        print(f"✓ Enhancement complete")
        
        return StreamingResponse(
            img_byte_arr, 
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=enhanced_{file.filename}"
            }
        )
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/diagnose")
async def diagnose_output(file: UploadFile = File(...)):
    """
    Diagnose raw model output WITHOUT any post-processing
    Use this to check if blur is from model or post-processing
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        input_tensor, original_size = preprocess_image(input_image)
        
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        print(f"📊 Diagnostic Info:")
        print(f"   Input range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
        print(f"   Output range: [{output_tensor.min():.3f}, {output_tensor.max():.3f}]")
        
        # RAW output with ONLY range conversion
        raw_image = no_postprocess(output_tensor, original_size)
        
        img_byte_arr = io.BytesIO()
        raw_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(
            img_byte_arr,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=raw_{file.filename}"}
        )
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare")
async def compare_images(file: UploadFile = File(...)):
    """Return both original and enhanced images"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
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
            "original_size": original_size,
            "model_output_range": [float(output_tensor.min()), float(output_tensor.max())],
            "config": CONFIG
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🌊 Starting AquaEnhance FastAPI Server")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")