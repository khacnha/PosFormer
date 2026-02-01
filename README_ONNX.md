# PosFormer ONNX Export & Web Demo

This document describes how to export the PosFormer model to ONNX and run inference in the browser.

## Prerequisites

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install pytorch-lightning==1.4.9 torchmetrics==0.6.0 einops opencv-python-headless pillow onnx onnxruntime
```

## Changes Made for ONNX Compatibility

### 1. MaskBatchNorm2d Patch (`Pos_Former/model/transformer/arm.py`)

The original code uses boolean indexing (`x[mask]`) which creates dynamic shapes incompatible with ONNX. The patch applies BatchNorm to the entire tensor then masks the result:

```python
# Original (training mode preserved):
flat_x = x[not_mask, :]  # Dynamic indexing - breaks ONNX

# Patched (inference mode):
out = F.batch_norm(x, ...)  # Apply to full tensor
out = out.masked_fill(mask, 0.0)  # Mask invalid regions
```

### 2. Export Script (`scripts/export_onnx.py`)

- Monkeypatches `torch.load` to handle `weights_only=False` for legacy checkpoints
- Exports Encoder and Decoder as separate ONNX models with dynamic axes
- Wraps Decoder to return only last token logits for autoregressive decoding

### 3. Web Preprocessing (`web/utils.js`)

- **No color inversion** - expects black background, white text images
- Crops to content, scales to model size limits (16-256 height, 16-1024 width)

## Export Commands

```bash
# Activate environment
source .venv/bin/activate

# Convert vocabulary to JavaScript
python scripts/convert_vocab.py

# Export ONNX models
PYTHONPATH=. python scripts/export_onnx.py

# Move models to web directory
mv web/public/models/* web/models/
rm -rf web/public
```

## Test ONNX Inference (Python)

```bash
PYTHONPATH=. python scripts/test_onnx_inference.py inverted_image.png
# Expected output: "2 + x"
```

## Run Web Demo

```bash
cd web
python -m http.server 8000
# Open http://localhost:8000
```

## ⚠️ Image Format Requirement

**Images must have BLACK background and WHITE text/formula.**

To invert a white-background image:
```python
from PIL import Image
import numpy as np
img = Image.open('input.png').convert('L')
Image.fromarray(255 - np.array(img)).save('inverted.png')
```

## File Structure

```
PosFormer/
├── scripts/
│   ├── export_onnx.py      # ONNX export script
│   ├── convert_vocab.py    # Vocabulary converter
│   └── test_onnx_inference.py  # Python ONNX test
├── web/
│   ├── models/
│   │   ├── encoder.onnx
│   │   └── decoder.onnx
│   ├── index.html
│   ├── style.css
│   ├── model.js            # ONNX inference loop
│   ├── utils.js            # Image preprocessing
│   ├── vocab.js            # Token vocabulary
│   └── main.js             # UI controller
└── Pos_Former/model/transformer/
    └── arm.py              # Patched MaskBatchNorm2d
```
