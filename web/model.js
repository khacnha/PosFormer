
let encoderSession = null;
let decoderSession = null;

const MAX_LEN = 150; // reasonable limit for formula

async function loadModels() {
    updateStatus("Loading models...");
    try {
        // Disable SIMD and Threading for stability
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.simd = false;

        const options = { 
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        };
        encoderSession = await ort.InferenceSession.create('./models/encoder.onnx', options);
        decoderSession = await ort.InferenceSession.create('./models/decoder.onnx', options);
        updateStatus("Models loaded. Ready.");
        return true;
    } catch (e) {
        console.error(e);
        updateStatus("Error loading models: " + e.message);
        return false;
    }
}

async function predictFormula(imageElement) {
    if (!encoderSession || !decoderSession) {
        alert("Models not loaded yet.");
        return;
    }

    updateStatus("Preprocessing image...");
    try {
        // 1. Preprocess
        const { tensor, mask, dims } = await preprocessImage(imageElement);
        
        // Debug: Show processed image
        // document.body.appendChild(processedImageCanvas);

        updateStatus("Running Encoder...");
        
        // 2. Encoder
        const encoderFeeds = { 
            img: tensor,
            img_mask: mask 
        };
        const encoderResults = await encoderSession.run(encoderFeeds);
        const feature = encoderResults.feature; // [1, seq_len, d_model]
        const featureMask = encoderResults.mask; // [1, seq_len]

        updateStatus("Running Decoder...");

        // 3. Decoder Loop (Greedy Search)
        // Initial tgt: [SOS]
        let tgtIndices = [VOCAB.SOS_IDX]; // [1]
        
        // We will run step by step
        for (let i = 0; i < MAX_LEN; i++) {
            // Prepare inputs
            // feature, enc_mask are constant
            // tgt changes
            
            // Re-create tensors to ensure clean data transfer between sessions
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
            // Shape: [1, 1, vocab_size] (because we extracted last token logits in export)
            // Actually export wrapper returns out[:, -1, :], so shape [batch, vocab_size] -> [1, vocab_size]
            
            const outputData = logits.data; // Float32Array
            
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
            
            // Optional: update UI live?
            // updateStatus(`Decoding... ${i}/${MAX_LEN}`);
        }
        
        // 4. Decode Indices to String
        // Remove SOS
        const resultIndices = tgtIndices.slice(1);
        const formula = resultIndices.map(idx => VOCAB.idx2word[idx]).join(' ');
        
        return formula;

    } catch (e) {
        console.error(e);
        updateStatus("Error during prediction: " + e.message);
        throw e;
    }
}

function updateStatus(msg) {
    const el = document.getElementById('status');
    if (el) el.textContent = "Status: " + msg;
}
