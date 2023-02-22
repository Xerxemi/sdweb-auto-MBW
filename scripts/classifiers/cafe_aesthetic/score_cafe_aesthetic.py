import os
import torch
import safetensors
from transformers import CLIPModel, CLIPProcessor
from scripts.classifiers.cafe_aesthetic.aesthetic import judge

def score(image):
    aesthetic, _, _ = judge(image)
    return aesthetic["aesthetic"]

