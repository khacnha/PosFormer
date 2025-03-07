from Pos_Former.lit_posformer import LitPosFormer
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from Pos_Former.datamodule.transforms import ScaleToLimitRange
from Pos_Former.datamodule import vocab

def load_model(checkpoint_path):
    """load model from checkpoint and set to evaluation mode"""
    model = LitPosFormer.load_from_checkpoint(checkpoint_path)
    model.eval()  # set model to evaluation mode
    return model

def crop_image(image_np, thresh=240, padding=5):
    """crop the extra background, only keep the content"""
    # find the pixels that are not background (assume background is white, high value)
    content_mask = image_np < thresh

    # find the position of the content pixels
    rows = np.any(content_mask, axis=1)
    cols = np.any(content_mask, axis=0)
    if np.any(rows) and np.any(cols):  # ensure there is content in the image
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # add a small margin
        rmin = max(0, rmin - padding)
        rmax = min(image_np.shape[0] - 1, rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(image_np.shape[1] - 1, cmax + padding)
        
        # crop the image
        cropped_image = image_np[rmin:rmax+1, cmin:cmax+1]
        return cropped_image, True
    else:
        print("no content found in the image")
        return image_np, False

def preprocess_image(image_path, save_intermediate=True):
    """read and preprocess the image"""
    # read the image and convert to grayscale
    image = Image.open(image_path).convert('L')
    image_np = np.array(image)
    
    # crop the extra background
    image_np, success = crop_image(image_np)
    if success and save_intermediate:
        Image.fromarray(image_np).save("cropped_image.png")
    
    # invert the colors: white background, black text -> black background, white text
    image_np = 255 - image_np  
    if save_intermediate:
        Image.fromarray(image_np).save("inverted_image.png")
    
    # apply the same transformations as in the evaluation process
    transform = transforms.Compose([
        ScaleToLimitRange(w_lo=16, w_hi=1024, h_lo=16, h_hi=256),
        transforms.ToTensor(),
    ])
    
    # process the image
    img_tensor = transform(image_np).unsqueeze(0)  # add batch dimension [1, 1, H, W]
    img_mask = torch.zeros(1, img_tensor.shape[2], img_tensor.shape[3], dtype=torch.bool)
    
    return img_tensor, img_mask

def predict(model, img_tensor, img_mask, device):
    """perform prediction on the preprocessed image"""
    # move the model to the computing device
    model = model.to(device)
    img_tensor = img_tensor.to(device)
    img_mask = img_mask.to(device)
    
    # perform prediction
    with torch.no_grad():
        hypotheses = model.approximate_joint_search(img_tensor, img_mask)
        
        # get the best prediction
        best_hypothesis = hypotheses[0]
        predicted_indices = best_hypothesis.seq
        
        # convert indices to LaTeX expression
        predicted_latex = vocab.indices2label(predicted_indices)
        
    return predicted_latex

def main():
    # path to the checkpoint file
    checkpoint_path = "./lightning_logs/version_0/checkpoints/best.ckpt"
    
    # path to the image file
    image_path = "./image.png"
    
    # computing device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load the model
    model = load_model(checkpoint_path)
    
    # preprocess the image
    img_tensor, img_mask = preprocess_image(image_path)
    
    # predict
    predicted_latex = predict(model, img_tensor, img_mask, device)
    
    # display the result
    print(f"input: {image_path}")
    print(f"recognized expression: ${predicted_latex}$")

if __name__ == "__main__":
    main()