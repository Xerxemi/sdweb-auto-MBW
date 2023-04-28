import os
import torch
import math
import ImageReward as reward

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

model = None
def score(image, prompt=""):
    global model
    if model == None:
        model = reward.load("ImageReward-v1.0")
    score_origin = model.score(prompt, image)
    score = sigmoid(score_origin)
    return score