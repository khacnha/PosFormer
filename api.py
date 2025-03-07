from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import tempfile
import os
import shutil
from PIL import Image
from typing import Optional
from predict import load_model, preprocess_image, predict

app = FastAPI(
    title="PosFormer API",
    description="API to recognize mathematical formulas from images",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load the model when the application starts
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = "./lightning_logs/version_0/checkpoints/best.ckpt"
model = None

@app.on_event("startup")
async def startup_event():
    global model
    print(f"Loading model from {checkpoint_path}...")
    try:
        model = load_model(checkpoint_path)
        print(f"Model loaded successfully. Using device: {device}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model = None

@app.post("/predict/", response_class=JSONResponse)
async def predict_formula(file: UploadFile = File(...), save_intermediate: Optional[bool] = False):
    """
    API endpoint to recognize mathematical formulas from images
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Check if the file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted")
    
    try:
        # Save the temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
            shutil.copyfileobj(file.file, temp)
            temp_path = temp.name
        
        # Preprocess the image
        img_tensor, img_mask = preprocess_image(temp_path, save_intermediate=save_intermediate)
        
        # Predict
        latex_formula = predict(model, img_tensor, img_mask, device)
        
        # Delete the temporary file
        os.unlink(temp_path)
        
        # Return the result
        return {
            "success": True,
            "formula": latex_formula,
            "formula_latex": f"${latex_formula}$"
        }
    
    except Exception as e:
        # Ensure the temporary file is deleted if there is an error
        if 'temp_path' in locals():
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Error processing: {str(e)}")

@app.get("/health")
def health_check():
    """Check the status of the API"""
    return {"status": "healthy", "model_loaded": model is not None}

# if __name__ == "__main__":
#     uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
