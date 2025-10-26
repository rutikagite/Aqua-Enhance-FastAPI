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

# Import your CLCC model
from model import CLCC  # Make sure model.py is in the same directory

app = FastAPI(title="AquaEnhance API", version="1.0.0")

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Streamlit URL
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
    'image_size': 256  # Adjust based on your training
}

def load_model(model_path):
    """
    Load the trained CLCC model
    """
    try:
        # Initialize model with your architecture
        model = CLCC(
            channel_scale=CONFIG['channel_scale'],
            main_ks=CONFIG['main_ks'],
            gcc_ks=CONFIG['gcc_ks']
        )
        
        # Load the trained weights
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        print(f"Model loaded successfully on {device}")
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model
    try:
        # Update this path to your trained model
        model_path = "checkpoints/best_model.pth"  # or "model_best.pth"
        
        if not Path(model_path).exists():
            print(f"Warning: Model file not found at {model_path}")
            print("Please update the model_path variable with the correct path")
            return
        
        model = load_model(model_path)
        print("✓ AquaEnhance model ready!")
        
    except Exception as e:
        print(f"✗ Error during startup: {e}")
        print("The API will still run but enhancement won't work until model is loaded")

def preprocess_image(image: Image.Image):
    """
    Preprocess image for CLCC model input
    - Resize to training size
    - Convert to tensor
    - Normalize to [0, 1] range
    """
    # Get original size to restore later
    original_size = image.size
    
    # Resize to model input size
    transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W)
    ])
    
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return img_tensor.to(device), original_size

def postprocess_output(output_tensor, original_size):
    """
    Convert model output back to image
    - Handle Tanh output range [-1, 1]
    - Resize back to original dimensions
    - Convert to PIL Image
    """
    # Remove batch dimension and move to CPU
    output = output_tensor.squeeze(0).cpu().detach()
    
    # CLCC uses Tanh activation, so output is in [-1, 1]
    # Convert to [0, 1] range
    output = (output + 1) / 2.0
    
    # Clip to ensure valid range
    output = torch.clamp(output, 0, 1)
    
    # Convert to PIL Image
    output_np = output.numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    output_np = (output_np * 255).astype(np.uint8)
    
    enhanced_image = Image.fromarray(output_np)
    
    # Resize back to original dimensions
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
        "config": CONFIG
    }

@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    """
    Enhance underwater image using CLCC model
    
    Args:
        file: Uploaded image file (JPG, JPEG, PNG)
        
    Returns:
        Enhanced image as PNG
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs and ensure model file exists."
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type: {file.content_type}. Please upload an image."
        )
    
    try:
        # Read uploaded file
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        print(f"Processing image: {file.filename} | Size: {input_image.size}")
        
        # Preprocess image
        input_tensor, original_size = preprocess_image(input_image)
        
        # Run inference
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        # Postprocess output
        enhanced_image = postprocess_output(output_tensor, original_size)
        
        # Convert to bytes for response
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
        print(f"✗ Error processing {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing image: {str(e)}"
        )

@app.post("/enhance-batch")
async def enhance_batch(files: list[UploadFile] = File(...)):
    """
    Enhance multiple underwater images
    
    Args:
        files: List of uploaded image files (max 10)
        
    Returns:
        Success message with count of processed images
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(
            status_code=400, 
            detail="Maximum 10 images per batch. Please reduce the number of files."
        )
    
    results = []
    errors = []
    
    for idx, file in enumerate(files):
        try:
            contents = await file.read()
            input_image = Image.open(io.BytesIO(contents)).convert('RGB')
            
            input_tensor, original_size = preprocess_image(input_image)
            
            with torch.no_grad():
                output_tensor = model(input_tensor)
            
            enhanced_image = postprocess_output(output_tensor, original_size)
            
            # Store result
            img_byte_arr = io.BytesIO()
            enhanced_image.save(img_byte_arr, format='PNG', quality=95)
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "index": idx
            })
            
            print(f"✓ [{idx+1}/{len(files)}] Enhanced: {file.filename}")
            
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e),
                "index": idx
            })
            print(f"✗ [{idx+1}/{len(files)}] Failed: {file.filename} - {str(e)}")
    
    return {
        "message": "Batch processing complete",
        "total_files": len(files),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors if errors else None
    }

@app.post("/compare")
async def compare_images(file: UploadFile = File(...)):
    """
    Return both original and enhanced images for side-by-side comparison
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON with base64 encoded original and enhanced images
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import base64
        
        # Read uploaded file
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess and enhance
        input_tensor, original_size = preprocess_image(input_image)
        
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        enhanced_image = postprocess_output(output_tensor, original_size)
        
        # Convert both images to base64
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
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("🌊 AquaEnhance FastAPI Server")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Model: CLCC (Channel Scale: {CONFIG['channel_scale']})")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")