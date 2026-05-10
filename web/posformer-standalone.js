/**
 * PosFormer Standalone - Simple JavaScript library for formula recognition
 * Usage:
 *   <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
 *   <script src="posformer-standalone.js"></script>
 *   <script>
 *     const image = new Image();
 *     image.src = 'path/to/image.png';
 *     image.onload = async () => {
 *       const latex = await PosFormer.predict(image);
 *       console.log(latex);
 *     };
 *   </script>
 */

(function() {
    'use strict';

    // Vocabulary
    const VOCAB = {
        word2idx: {'<pad>': 0, '<sos>': 1, '<eos>': 2, '!': 3, '(': 4, ')': 5, '+': 6, ',': 7, '-': 8, '.': 9, '/': 10, '0': 11, '1': 12, '2': 13, '3': 14, '4': 15, '5': 16, '6': 17, '7': 18, '8': 19, '9': 20, '<': 21, '=': 22, '>': 23, 'A': 24, 'B': 25, 'C': 26, 'E': 27, 'F': 28, 'G': 29, 'H': 30, 'I': 31, 'L': 32, 'M': 33, 'N': 34, 'P': 35, 'R': 36, 'S': 37, 'T': 38, 'V': 39, 'X': 40, 'Y': 41, '[': 42, '\\Delta': 43, '\\Pi': 44, '\\alpha': 45, '\\beta': 46, '\\cdot': 47, '\\cdots': 48, '\\cos': 49, '\\div': 50, '\\exists': 51, '\\forall': 52, '\\frac': 53, '\\gamma': 54, '\\geq': 55, '\\in': 56, '\\infty': 57, '\\int': 58, '\\lambda': 59, '\\ldots': 60, '\\leq': 61, '\\lim': 62, '\\limits': 63, '\\log': 64, '\\mu': 65, '\\neq': 66, '\\phi': 67, '\\pi': 68, '\\pm': 69, '\\prime': 70, '\\rightarrow': 71, '\\sigma': 72, '\\sin': 73, '\\sqrt': 74, '\\sum': 75, '\\tan': 76, '\\theta': 77, '\\times': 78, '\\{': 79, '\\}': 80, ']': 81, '^': 82, '_': 83, 'a': 84, 'b': 85, 'c': 86, 'd': 87, 'e': 88, 'f': 89, 'g': 90, 'h': 91, 'i': 92, 'j': 93, 'k': 94, 'l': 95, 'm': 96, 'n': 97, 'o': 98, 'p': 99, 'q': 100, 'r': 101, 's': 102, 't': 103, 'u': 104, 'v': 105, 'w': 106, 'x': 107, 'y': 108, 'z': 109, '{': 110, '|': 111, '}': 112},
        idx2word: {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '!', 4: '(', 5: ')', 6: '+', 7: ',', 8: '-', 9: '.', 10: '/', 11: '0', 12: '1', 13: '2', 14: '3', 15: '4', 16: '5', 17: '6', 18: '7', 19: '8', 20: '9', 21: '<', 22: '=', 23: '>', 24: 'A', 25: 'B', 26: 'C', 27: 'E', 28: 'F', 29: 'G', 30: 'H', 31: 'I', 32: 'L', 33: 'M', 34: 'N', 35: 'P', 36: 'R', 37: 'S', 38: 'T', 39: 'V', 40: 'X', 41: 'Y', 42: '[', 43: '\\Delta', 44: '\\Pi', 45: '\\alpha', 46: '\\beta', 47: '\\cdot', 48: '\\cdots', 49: '\\cos', 50: '\\div', 51: '\\exists', 52: '\\forall', 53: '\\frac', 54: '\\gamma', 55: '\\geq', 56: '\\in', 57: '\\infty', 58: '\\int', 59: '\\lambda', 60: '\\ldots', 61: '\\leq', 62: '\\lim', 63: '\\limits', 64: '\\log', 65: '\\mu', 66: '\\neq', 67: '\\phi', 68: '\\pi', 69: '\\pm', 70: '\\prime', 71: '\\rightarrow', 72: '\\sigma', 73: '\\sin', 74: '\\sqrt', 75: '\\sum', 76: '\\tan', 77: '\\theta', 78: '\\times', 79: '\\{', 80: '\\}', 81: ']', 82: '^', 83: '_', 84: 'a', 85: 'b', 86: 'c', 87: 'd', 88: 'e', 89: 'f', 90: 'g', 91: 'h', 92: 'i', 93: 'j', 94: 'k', 95: 'l', 96: 'm', 97: 'n', 98: 'o', 99: 'p', 100: 'q', 101: 'r', 102: 's', 103: 't', 104: 'u', 105: 'v', 106: 'w', 107: 'x', 108: 'y', 109: 'z', 110: '{', 111: '|', 112: '}'},
        PAD_IDX: 0,
        SOS_IDX: 1,
        EOS_IDX: 2
    };

    // Constants
    const W_LO = 16;
    const W_HI = 1024;
    const H_LO = 16;
    const H_HI = 256;
    const PADDING = 5;
    const THRESH = 240;
    const MAX_LEN = 150;

    // Model state
    let encoderSession = null;
    let decoderSession = null;
    let isModelLoaded = false;
    let isLoading = false;
    let loadPromise = null;

    // Configuration
    const config = {
        encoderPath: './models/encoder.onnx',
        decoderPath: './models/decoder.onnx'
    };

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
        let grayData = new Uint8Array(width * height);
        let top = height, bottom = -1, left = width, right = -1;
        let hasContent = false;

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * 4;
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

        // 4. ScaleToLimitRange
        let finalWidth = croppedWidth;
        let finalHeight = croppedHeight;
        let finalGray = croppedGray;

        let scale_r = 1.0;
        if (Math.min(H_HI / finalHeight, W_HI / finalWidth) < 1.0) {
            scale_r = Math.min(H_HI / finalHeight, W_HI / finalWidth);
        } else if (Math.max(H_LO / finalHeight, W_LO / finalWidth) > 1.0) {
            scale_r = Math.max(H_LO / finalHeight, W_LO / finalWidth);
        }
        
        if (scale_r !== 1.0) {
            finalWidth = Math.round(croppedWidth * scale_r);
            finalHeight = Math.round(croppedHeight * scale_r);
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = croppedWidth;
            tempCanvas.height = croppedHeight;
            const tempCtx = tempCanvas.getContext('2d');
            const tempImgData = tempCtx.createImageData(croppedWidth, croppedHeight);
            for(let i=0; i<croppedGray.length; i++) {
                const val = croppedGray[i];
                tempImgData.data[i*4] = val;
                tempImgData.data[i*4+1] = val;
                tempImgData.data[i*4+2] = val;
                tempImgData.data[i*4+3] = 255;
            }
            tempCtx.putImageData(tempImgData, 0, 0);

            const resizeCanvas = document.createElement('canvas');
            resizeCanvas.width = finalWidth;
            resizeCanvas.height = finalHeight;
            const resizeCtx = resizeCanvas.getContext('2d');
            resizeCtx.imageSmoothingEnabled = true;
            resizeCtx.imageSmoothingQuality = 'high';
            resizeCtx.drawImage(tempCanvas, 0, 0, finalWidth, finalHeight);

            const resizedData = resizeCtx.getImageData(0, 0, finalWidth, finalHeight).data;
            finalGray = new Uint8Array(finalWidth * finalHeight);
            for (let i = 0; i < finalWidth * finalHeight; i++) {
                finalGray[i] = resizedData[i * 4];
            }
        }

        // 5. To Tensor [1, 1, H, W] and normalize to [0, 1]
        const floatData = new Float32Array(finalWidth * finalHeight);
        for (let i = 0; i < finalGray.length; i++) {
            floatData[i] = finalGray[i] / 255.0;
        }

        const tensor = new ort.Tensor('float32', floatData, [1, 1, finalHeight, finalWidth]);
        const maskData = new Uint8Array(finalHeight * finalWidth).fill(0);
        const maskTensor = new ort.Tensor('bool', maskData, [1, finalHeight, finalWidth]);

        return {
            tensor: tensor,
            mask: maskTensor,
            dims: [finalHeight, finalWidth]
        };
    }

    /**
     * Load ONNX models
     * @param {string} encoderPath - Path to encoder.onnx
     * @param {string} decoderPath - Path to decoder.onnx
     * @returns {Promise<boolean>}
     */
    async function loadModels(encoderPath, decoderPath) {
        if (isModelLoaded) {
            return true;
        }

        if (isLoading) {
            return loadPromise;
        }

        isLoading = true;
        loadPromise = (async () => {
            try {
                if (typeof ort === 'undefined') {
                    throw new Error('ONNX Runtime Web is not loaded. Please include: <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>');
                }

                // Disable SIMD and Threading for stability
                ort.env.wasm.numThreads = 1;
                ort.env.wasm.simd = false;

                const options = { 
                    executionProviders: ['wasm'],
                    graphOptimizationLevel: 'all'
                };

                const encoderUrl = encoderPath || config.encoderPath;
                const decoderUrl = decoderPath || config.decoderPath;

                encoderSession = await ort.InferenceSession.create(encoderUrl, options);
                decoderSession = await ort.InferenceSession.create(decoderUrl, options);
                
                isModelLoaded = true;
                isLoading = false;
                return true;
            } catch (e) {
                isLoading = false;
                console.error('Error loading models:', e);
                throw e;
            }
        })();

        return loadPromise;
    }

    /**
     * Predict LaTeX formula from image
     * @param {HTMLImageElement|string|File} image - Image element, image URL, or File object
     * @returns {Promise<string>} LaTeX formula
     */
    async function predict(image) {
        // Ensure models are loaded
        if (!isModelLoaded) {
            await loadModels();
        }

        if (!encoderSession || !decoderSession) {
            throw new Error("Models not loaded yet.");
        }

        // Handle different input types
        let imageElement;
        if (typeof image === 'string') {
            // URL string
            imageElement = new Image();
            await new Promise((resolve, reject) => {
                imageElement.onload = resolve;
                imageElement.onerror = reject;
                imageElement.src = image;
            });
        } else if (image instanceof File) {
            // File object
            imageElement = new Image();
            const url = URL.createObjectURL(image);
            await new Promise((resolve, reject) => {
                imageElement.onload = () => {
                    URL.revokeObjectURL(url);
                    resolve();
                };
                imageElement.onerror = () => {
                    URL.revokeObjectURL(url);
                    reject(new Error('Failed to load image file'));
                };
                imageElement.src = url;
            });
        } else if (image instanceof HTMLImageElement) {
            // Already an image element
            imageElement = image;
            if (!imageElement.complete) {
                await new Promise((resolve, reject) => {
                    imageElement.onload = resolve;
                    imageElement.onerror = reject;
                });
            }
        } else {
            throw new Error('Invalid image input. Expected HTMLImageElement, URL string, or File object.');
        }

        try {
            // 1. Preprocess
            const { tensor, mask } = await preprocessImage(imageElement);
            
            // 2. Encoder
            const encoderFeeds = { 
                img: tensor,
                img_mask: mask 
            };
            const encoderResults = await encoderSession.run(encoderFeeds);
            const feature = encoderResults.feature;
            const featureMask = encoderResults.mask;

            // 3. Decoder Loop (Greedy Search)
            let tgtIndices = [VOCAB.SOS_IDX];
            
            for (let i = 0; i < MAX_LEN; i++) {
                const featureTensor = new ort.Tensor(feature.type, feature.data, feature.dims);
                const maskTensor = new ort.Tensor(featureMask.type, featureMask.data, featureMask.dims);
                const tgtTensor = new ort.Tensor('int64', new BigInt64Array(tgtIndices.map(x => BigInt(x))), [1, tgtIndices.length]);
                
                const decoderFeeds = {
                    feature: featureTensor,
                    enc_mask: maskTensor,
                    tgt: tgtTensor
                };
                
                const decoderResults = await decoderSession.run(decoderFeeds);
                const logits = decoderResults.logits;
                const outputData = logits.data;
                
                // Argmax
                let maxVal = -Infinity;
                let maxIdx = -1;
                for (let j = 0; j < outputData.length; j++) {
                    if (outputData[j] > maxVal) {
                        maxVal = outputData[j];
                        maxIdx = j;
                    }
                }
                
                if (maxIdx === VOCAB.EOS_IDX) {
                    break;
                }
                
                tgtIndices.push(maxIdx);
            }
            
            // 4. Decode Indices to String
            const resultIndices = tgtIndices.slice(1);
            const formula = resultIndices.map(idx => VOCAB.idx2word[idx]).join(' ');
            
            return formula;

        } catch (e) {
            console.error('Error during prediction:', e);
            throw e;
        }
    }

    // Auto-load models when script is loaded
    if (typeof window !== 'undefined') {
        // Wait for DOM and ONNX Runtime to be ready
        const autoLoad = () => {
            if (typeof ort !== 'undefined') {
                loadModels().catch(err => {
                    console.warn('Auto-load models failed:', err);
                });
            } else {
                // Retry after a short delay
                setTimeout(autoLoad, 100);
            }
        };

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', autoLoad);
        } else {
            autoLoad();
        }
    }

    // Export API
    const PosFormer = {
        predict: predict,
        loadModels: loadModels,
        isLoaded: () => isModelLoaded,
        setConfig: (newConfig) => {
            if (newConfig.encoderPath) config.encoderPath = newConfig.encoderPath;
            if (newConfig.decoderPath) config.decoderPath = newConfig.decoderPath;
        }
    };

    // Export to global scope
    if (typeof window !== 'undefined') {
        window.PosFormer = PosFormer;
    }
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = PosFormer;
    }

})();
