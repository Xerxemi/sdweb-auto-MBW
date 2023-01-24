import torch
import numpy as np

use_cuda = torch.cuda.is_available()

def image_embeddings_direct(image, model, processor):
    inputs = processor(images=image, return_tensors='pt')['pixel_values']
    if use_cuda:
        inputs = inputs.to('cuda')
    result = model.get_image_features(pixel_values=inputs).cpu().detach().numpy()
    return (result / np.linalg.norm(result)).squeeze(axis=0)

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
