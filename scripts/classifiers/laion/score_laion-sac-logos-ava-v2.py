import os
import torch
import safetensors
from scripts.classifiers.laion.laion import image_embeddings_direct_laion, MLP

dirname = os.path.dirname(__file__)
aesthetic_path = os.path.join(dirname, "laion-sac-logos-ava-v2.safetensors")
aes_model = MLP(768).to('cuda').eval()
aes_model.load_state_dict(safetensors.torch.load_file(aesthetic_path))

def score(image):
    image_embeds = image_embeddings_direct_laion(image)
    prediction = aes_model(torch.from_numpy(image_embeds).float().to('cuda'))
    return prediction.item()

