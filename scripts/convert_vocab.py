import os

def convert_vocab():
    dict_path = "Pos_Former/datamodule/dictionary.txt"
    js_output_path = "web/vocab.js"
    
    word2idx = {}
    word2idx["<pad>"] = 0
    word2idx["<sos>"] = 1
    word2idx["<eos>"] = 2
    
    print(f"Reading dictionary from {dict_path}")
    with open(dict_path, "r") as f:
        for line in f.readlines():
            w = line.strip()
            word2idx[w] = len(word2idx)
            
    idx2word = {v: k for k, v in word2idx.items()}
    
    print(f"Vocab size: {len(word2idx)}")
    
    # Generate JS file
    js_content = f"""
const VOCAB = {{
    word2idx: {word2idx},
    idx2word: {idx2word},
    PAD_IDX: 0,
    SOS_IDX: 1,
    EOS_IDX: 2
}};
"""
    with open(js_output_path, "w") as f:
        f.write(js_content)
        
    print(f"Saved vocabulary to {js_output_path}")

if __name__ == "__main__":
    convert_vocab()
