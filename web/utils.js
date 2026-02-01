
// Constants matching predict.py
const W_LO = 16;
const W_HI = 1024;
const H_LO = 16;
const H_HI = 256;
const PADDING = 5;
const THRESH = 240;

/**
 * Preprocess image for PosFormer model
 * @param {HTMLImageElement} imageElement 
 * @returns {Promise<{tensor: ort.Tensor, mask: ort.Tensor, dims: [number, number]}>}
 */
async function preprocessImage(imageElement) {
    // 1. Draw to canvas to get pixel data
    const canvas = document.createElement('canvas');
    canvas.width = imageElement.width;
    canvas.height = imageElement.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0);
    
    let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let data = imageData.data; // RGBA
    let width = canvas.width;
    let height = canvas.height;

    // 2. Convert to Grayscale & Check content
    // Also find bounds for cropping
    let grayData = new Uint8Array(width * height);
    let top = height, bottom = -1, left = width, right = -1;
    let hasContent = false;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4;
            // Grayscale: 0.299R + 0.587G + 0.114B
            const gray = data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114;
            grayData[y * width + x] = gray;

            if (gray < THRESH) {
                hasContent = true;
                if (y < top) top = y;
                if (y > bottom) bottom = y;
                if (x < left) left = x;
                if (x > right) right = x;
            }
        }
    }

    if (!hasContent) {
        throw new Error("No content found in image");
    }

    // 3. Crop
    top = Math.max(0, top - PADDING);
    bottom = Math.min(height - 1, bottom + PADDING);
    left = Math.max(0, left - PADDING);
    right = Math.min(width - 1, right + PADDING);

    const croppedWidth = right - left + 1;
    const croppedHeight = bottom - top + 1;
    let croppedGray = new Uint8Array(croppedWidth * croppedHeight);

    for (let y = 0; y < croppedHeight; y++) {
        for (let x = 0; x < croppedWidth; x++) {
            croppedGray[y * croppedWidth + x] = grayData[(top + y) * width + (left + x)];
        }
    }

    // 4. NO INVERSION - User provides images with black background, white text already
    // (Model expects: black bg = 0, white text = high values, then normalized to [0,1])

    // 5. ScaleToLimitRange
    let finalWidth = croppedWidth;
    let finalHeight = croppedHeight;
    let finalGray = croppedGray;

    const r = finalHeight / finalWidth;
    const lo_r = H_LO / W_HI;
    const hi_r = H_HI / W_LO;

    // Only scale if ratio is within reasonable bounds (assert in python)
    // Python code asserts: lo_r <= r <= hi_r. If not, python raises error.
    // We should probably just best-effort scale or warn.
    
    let scale_r = 1.0;
    if (Math.min(H_HI / finalHeight, W_HI / finalWidth) < 1.0) {
        scale_r = Math.min(H_HI / finalHeight, W_HI / finalWidth);
    } else if (Math.max(H_LO / finalHeight, W_LO / finalWidth) > 1.0) {
        scale_r = Math.max(H_LO / finalHeight, W_LO / finalWidth);
    }
    
    if (scale_r !== 1.0) {
        finalWidth = Math.round(croppedWidth * scale_r);
        finalHeight = Math.round(croppedHeight * scale_r);
        // Use canvas for resizing (bilinear/bicubic quality is handled by browser)
        // We need to put croppedGray back to canvas to resize
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = croppedWidth;
        tempCanvas.height = croppedHeight;
        const tempCtx = tempCanvas.getContext('2d');
        const tempImgData = tempCtx.createImageData(croppedWidth, croppedHeight);
        for(let i=0; i<croppedGray.length; i++) {
            const val = croppedGray[i]; // It's inverted now. 
            // Put it into RGBA. 
            // Wait, we want to resize the INVERTED image? Yes.
            tempImgData.data[i*4] = val;
            tempImgData.data[i*4+1] = val;
            tempImgData.data[i*4+2] = val;
            tempImgData.data[i*4+3] = 255;
        }
        tempCtx.putImageData(tempImgData, 0, 0);

        // Draw to resized canvas
        const resizeCanvas = document.createElement('canvas');
        resizeCanvas.width = finalWidth;
        resizeCanvas.height = finalHeight;
        const resizeCtx = resizeCanvas.getContext('2d');
        // Enhance settings
        resizeCtx.imageSmoothingEnabled = true;
        resizeCtx.imageSmoothingQuality = 'high';
        resizeCtx.drawImage(tempCanvas, 0, 0, finalWidth, finalHeight);

        // Read back
        const resizedData = resizeCtx.getImageData(0, 0, finalWidth, finalHeight).data;
        finalGray = new Uint8Array(finalWidth * finalHeight);
        for (let i = 0; i < finalWidth * finalHeight; i++) {
            // Take Red channel (grayscale)
            finalGray[i] = resizedData[i * 4];
        }
    }

    // 6. To Tensor [1, 1, H, W]
    // Normalize to [0, 1]
    const floatData = new Float32Array(finalWidth * finalHeight);
    for (let i = 0; i < finalGray.length; i++) {
        floatData[i] = finalGray[i] / 255.0;
    }

    // Return as ONNX Tensor
    const tensor = new ort.Tensor('float32', floatData, [1, 1, finalHeight, finalWidth]);
    
    // Mask [1, H, W] zeros (bool/long?)
    // In Python: torch.zeros(1, H, W, dtype=torch.bool)
    // ONNX Runtime Web usually expects int64 or bool tensor.
    // Export script dummy contained torch.long so likely int64. 
    // Wait, in export script I used `torch.zeros(..., dtype=torch.long)` for encoder mask?
    // Let me check export script... `dummy_mask = torch.zeros(1, 256, 1024, dtype=torch.long)`
    // So we should return int64 tensor of zeros.
    // Mask [1, H, W] zeros (bool)
    // ONNX Runtime Web expects bool tensors to be backed by Uint8Array (0 or 1)
    const maskData = new Uint8Array(finalHeight * finalWidth).fill(0);
    const maskTensor = new ort.Tensor('bool', maskData, [1, finalHeight, finalWidth]);

    return {
        tensor: tensor,
        mask: maskTensor,
        dims: [finalHeight, finalWidth],
        processedImageCanvas: drawToDebugCanvas(floatData, finalHeight, finalWidth)
    };
}

function drawToDebugCanvas(data, h, w) {
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(w, h);
    for (let i = 0; i < data.length; i++) {
        const val = data[i] * 255;
        imgData.data[i * 4] = val;
        imgData.data[i * 4 + 1] = val;
        imgData.data[i * 4 + 2] = val;
        imgData.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
    return canvas;
}
