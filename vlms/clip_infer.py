import clip
from PIL import Image
import torch
import numpy as np
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, preprocess = clip.load("ViT-L/14@336px", device=device)

def clip_infer(rgb1, rgb2, text):
    similarities = []
    text = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    for rgb in [rgb1, rgb2]:
        PIL_image = Image.fromarray(rgb).convert('RGB')
        image = preprocess(PIL_image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T)
            similarities.append(similarity.cpu().numpy()[0][0])
    
    if similarities[0] > similarities[1]:
        return 0
    elif similarities[0] < similarities[1]:
        return 1
    else:
        return -1
    
def clip_infer_score(rgb, text):
    text = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    PIL_image = Image.fromarray(rgb).convert('RGB')
    image = preprocess(PIL_image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T)

    similarity = similarity.cpu().numpy()[0][0]

    return similarity