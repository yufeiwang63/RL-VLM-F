import torch
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain_vitL", device=device, is_eval=True)


def blip2_infer_image_text_matching(rgb1, rgb2, text, use_prob=False, return_scores=False):

    caption = text
    with torch.no_grad():
        txt = text_processors["eval"](caption)

    matching_probabilities = []
    matching_cosine_scores = []
    for rgb in [rgb1, rgb2]:
        raw_image = Image.fromarray(rgb).convert("RGB")
        with torch.no_grad():
            img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

            itm_output = model({"image": img, "text_input": txt}, match_head="itm")
            itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
            itm_score = itm_scores[:, 1].item()
            matching_probabilities.append(itm_score)

            itc_score = model({"image": img, "text_input": txt}, match_head='itc')#.item()
            matching_cosine_scores.append(itc_score)
        
    if not return_scores:
        if use_prob:
            if matching_probabilities[0] > matching_probabilities[1]:
                return 0
            elif matching_probabilities[0] < matching_probabilities[1]:
                return 1
            else:
                return -1
        else:
            if matching_cosine_scores[0] > matching_cosine_scores[1]:
                return 0
            elif matching_cosine_scores[0] < matching_cosine_scores[1]:
                return 1
            else:
                return -1
    else:
        return matching_cosine_scores
    
def blip2_image_text_matching(rgb, text, use_prob=False):

    caption = text
    with torch.no_grad():
        txt = text_processors["eval"](caption)

        raw_image = Image.fromarray(rgb).convert("RGB")
        img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        itm_output = model({"image": img, "text_input": txt}, match_head="itm")
        itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
        itm_score = itm_scores[:, 1].item()
        itc_score = model({"image": img, "text_input": txt}, match_head='itc').item()
        
    if use_prob:
        return itm_score
    else:
        return itc_score
    