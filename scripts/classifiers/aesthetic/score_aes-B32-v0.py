import os
import torch
import safetensors
from transformers import CLIPModel, CLIPProcessor
from scripts.classifiers.aesthetic.aesthetic import image_embeddings_direct, Classifier

dirname = os.path.dirname(__file__)
aesthetic_path = os.path.join(dirname, "aes-B32-v0.safetensors")
clip_name = 'openai/clip-vit-base-patch32'
clipprocessor = CLIPProcessor.from_pretrained(clip_name)
clipmodel = CLIPModel.from_pretrained(clip_name).to('cuda').eval()
aes_model = Classifier(512, 256, 1).to('cuda')
aes_model.load_state_dict(safetensors.torch.load_file(aesthetic_path))

def score(image):
    image_embeds = image_embeddings_direct(image, clipmodel, clipprocessor)
    prediction = aes_model(torch.from_numpy(image_embeds).float().to('cuda'))
    return prediction.item()

