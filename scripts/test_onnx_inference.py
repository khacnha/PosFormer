import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())
from Pos_Former.datamodule import vocab

def preprocess(image_path):
    # Same logic as predict.py / utils.js
    thresh = 240
    padding = 5
    
    # 1. Read and grayscale
    image = Image.open(image_path).convert('L')
    image_np = np.array(image)
    
    # 2. Crop
    content_mask = image_np < thresh
    rows = np.any(content_mask, axis=1)
    cols = np.any(content_mask, axis=0)
    
    if np.any(rows) and np.any(cols):
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmin = max(0, rmin - padding)
        rmax = min(image_np.shape[0] - 1, rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(image_np.shape[1] - 1, cmax + padding)
        image_np = image_np[rmin:rmax+1, cmin:cmax+1]
    
    # 3. NO INVERSION - User provides images with black bg, white text already
    
    # 4. Resize (ScaleToLimitRange logic)
    h, w = image_np.shape
    w_lo, w_hi = 16, 1024
    h_lo, h_hi = 16, 256
    
    scale_r = 1.0
    if min(h_hi / h, w_hi / w) < 1.0:
        scale_r = min(h_hi / h, w_hi / w)
    elif max(h_lo / h, w_lo / w) > 1.0:
        scale_r = max(h_lo / h, w_lo / w)
        
    if scale_r != 1.0:
        image_np = cv2.resize(image_np, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_LINEAR)
        
    # 5. To Tensor [1, 1, H, W]
    image_np = image_np.astype(np.float32) / 255.0
    img_tensor = image_np[np.newaxis, np.newaxis, :, :] # [1, 1, H, W]
    
    # Mask [1, H, W]
    h, w = image_np.shape
    mask = np.zeros((1, h, w), dtype=bool)
    
    return img_tensor, mask

def run_inference(image_path):
    encoder_path = "./web/models/encoder.onnx"
    decoder_path = "./web/models/decoder.onnx"
    
    print(f"Loading models from {encoder_path} and {decoder_path}...")
    enc_sess = ort.InferenceSession(encoder_path)
    dec_sess = ort.InferenceSession(decoder_path)
    
    print(f"Preprocessing {image_path}...")
    img, mask = preprocess(image_path)
    
    print("Running Encoder...")
    enc_inputs = {'img': img, 'img_mask': mask}
    enc_outs = enc_sess.run(None, enc_inputs)
    feature = enc_outs[0] # [1, H, W, D] or similar depending on export
    enc_mask = enc_outs[1]
    
    print("Running Decoder loop...")
    tgt = [vocab.SOS_IDX]
    max_len = 150
    
    for i in range(max_len):
        tgt_tensor = np.array([tgt], dtype=np.int64) # [1, seq_len]
        
        dec_inputs = {
            'feature': feature,
            'enc_mask': enc_mask,
            'tgt': tgt_tensor
        }
        
        logits = dec_sess.run(None, dec_inputs)[0] 
        
        next_token_logits = logits[0]
        next_token = np.argmax(next_token_logits)
        
        if next_token == vocab.EOS_IDX:
            break
            
        tgt.append(next_token)
        
    formula = vocab.indices2label(tgt[1:])
    print(f"Result: {formula}")
    return formula

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_onnx_inference.py <image_path>")
    else:
        run_inference(sys.argv[1])
