import onnxruntime as ort
import numpy as np

def verify():
    print("Loading Decoder...")
    sess = ort.InferenceSession("./web/models/decoder.onnx")
    
    # Inputs matching the crash case in JS
    # Feature: [1, 16, 23, 256] float32
    # Mask: [1, 16, 23] bool
    # Tgt: [1, 1] int64
    
    feature = np.random.randn(1, 16, 23, 256).astype(np.float32)
    mask = np.zeros((1, 16, 23), dtype=bool)
    tgt = np.array([[1]], dtype=np.int64)
    
    print("Running Decoder Inference...")
    try:
        inputs = {
            'feature': feature,
            'enc_mask': mask,
            'tgt': tgt
        }
        outputs = sess.run(None, inputs)
        print("Success!")
        print("Output shape:", outputs[0].shape)
    except Exception as e:
        print("Failed!")
        print(e)

if __name__ == "__main__":
    verify()
