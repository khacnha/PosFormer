import torch
import torch.nn as nn
import os

# Monkeypatch torch.load to handle weights_only=True default in newer torch versions
# This is necessary because the checkpoint contains pytorch-lightning objects
_original_load = torch.load
def _safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _safe_load

from Pos_Former.lit_posformer import LitPosFormer
from Pos_Former.datamodule import vocab

class EncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, img, img_mask):
        feature, mask = self.encoder(img, img_mask)
        return feature, mask

class DecoderWrapper(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, feature, mask, tgt):
        # Decoder returns (out, weights)
        # We only need 'out' which is [batch, len, vocab_size]
        out, _ = self.decoder(feature, mask, tgt)
        return out[:, -1, :] # Return only the logits for the last token

def export_onnx():
    checkpoint_path = "./lightning_logs/version_0/checkpoints/best_v1.ckpt"
    output_dir = "./web/public/models" # Exporting directly to web public dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {checkpoint_path}...")
    model = LitPosFormer.load_from_checkpoint(checkpoint_path)
    model.eval()

    # --- Export Encoder ---
    print("Exporting Encoder...")
    encoder_wrapper = EncoderWrapper(model.model.encoder)
    
    # Dummy inputs for encoder
    # img: [batch_size, channels, H, W] -> PosFormer uses [1, 1, H, W] usually but dynamic size
    # Let's use a fixed representative size or small size, but define dynamic axes
    dummy_img = torch.randn(1, 1, 256, 1024) 
    # img_mask: [batch_size, H, W] -> bool.
    # Predict.py uses torch.bool. LitPosFormer type hint saying LongTensor is likely inaccurate or flexible.
    dummy_mask = torch.zeros(1, 256, 1024, dtype=torch.bool)

    torch.onnx.export(
        encoder_wrapper,
        (dummy_img, dummy_mask),
        os.path.join(output_dir, "encoder.onnx"),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['img', 'img_mask'],
        output_names=['feature', 'mask'],
        dynamic_axes={
            'img': {2: 'height', 3: 'width'},
            'img_mask': {1: 'height', 2: 'width'},
            'feature': {1: 'seq_len'},
            'mask': {1: 'seq_len'}
        }
    )
    print("Encoder exported.")

    # --- Export Decoder ---
    print("Exporting Decoder...")
    decoder_wrapper = DecoderWrapper(model.model.decoder)

    # Dummy inputs for decoder
    # feature: [batch, height, width, d_model] - output of encoder (DenseNet) is 4D
    # mask: [batch, height, width] - output of encoder mask
    # tgt: [batch, tgt_len] - current generated sequence
    
    d_model = model.hparams.d_model
    # Use representative size for H, W (e.g. 16x64 from 256x1024 input)
    dummy_feature = torch.randn(1, 16, 64, d_model)
    dummy_enc_mask = torch.zeros(1, 16, 64, dtype=torch.bool) 
    
    dummy_tgt = torch.tensor([[vocab.SOS_IDX]], dtype=torch.long)

    torch.onnx.export(
        decoder_wrapper,
        (dummy_feature, dummy_enc_mask, dummy_tgt),
        os.path.join(output_dir, "decoder.onnx"),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['feature', 'enc_mask', 'tgt'],
        output_names=['logits'],
        dynamic_axes={
            'feature': {1: 'height', 2: 'width'},
            'enc_mask': {1: 'height', 2: 'width'},
            'tgt': {1: 'tgt_len'}
        }
    )
    print("Decoder exported.")
    print(f"Models saved to {output_dir}")

if __name__ == "__main__":
    export_onnx()
