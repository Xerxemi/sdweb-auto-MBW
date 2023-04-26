import os
import torch
import ImageReward as reward

model = None
def score(image, prompt=""):
    global model
    if model == None:
        model = reward.load("ImageReward-v1.0")
    return model.score(prompt, image)
