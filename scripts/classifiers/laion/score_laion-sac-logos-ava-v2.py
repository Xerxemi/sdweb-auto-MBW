import os
import torch
import safetensors
from transformers import CLIPModel, CLIPProcessor
from scripts.classifiers.laion.laion import image_embeddings_direct, MLP

dirname = os.path.dirname(__file__)
aesthetic_path = os.path.join(dirname, "laion-sac-logos-ava-v2.safetensors")
clip_name = 'openai/clip-vit-large-patch14'
clipprocessor = CLIPProcessor.from_pretrained(clip_name)
clipmodel = CLIPModel.from_pretrained(clip_name).to('cuda').eval()
aes_model = MLP(768).to('cuda').eval()
aes_model.load_state_dict(safetensors.torch.load_file(aesthetic_path))

def score(image):
    image_embeds = image_embeddings_direct(image, clipmodel, clipprocessor)
    prediction = aes_model(torch.from_numpy(image_embeds).float().to('cuda'))
    return prediction.item()

