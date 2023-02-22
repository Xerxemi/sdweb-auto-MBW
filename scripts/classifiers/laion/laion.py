import torch
import numpy as np
import clip

use_cuda = torch.cuda.is_available()

def image_embeddings_direct(image, model, processor):
    inputs = processor(images=image, return_tensors='pt')['pixel_values']
    if use_cuda:
        inputs = inputs.to('cuda')
    result = model.get_image_features(pixel_values=inputs).cpu().detach().numpy()
    return (result / np.linalg.norm(result)).squeeze(axis=0)

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

def image_embeddings_direct_laion(pil_image):
    image = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    im_emb_arr = normalized(image_features.cpu().detach().numpy())
    return im_emb_arr

class MLP(torch.nn.Module):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)
