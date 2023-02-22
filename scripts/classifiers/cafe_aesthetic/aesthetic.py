import os
from pathlib import Path
from glob import glob
import shutil

import gradio as gr
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from webui import wrap_gradio_gpu_call
from modules import shared, scripts, script_callbacks, ui
from modules import generation_parameters_copypaste as parameters_copypaste
import launch

script_dir = Path(scripts.basedir())
aesthetics = {}  # name: pipeline


def library_check():
    if not launch.is_installed("transformers"):
        launch.run_pip("install transformers", "requirements for autoMBW: Cafe Aesthetic - adapted from p1atdev")


def model_check(name):
    if name not in aesthetics:
        library_check()
        from transformers import pipeline

        if name == "aesthetic":
            aesthetics["aesthetic"] = pipeline(
                "image-classification", model="cafeai/cafe_aesthetic"
            )
        elif name == "style":
            aesthetics["style"] = pipeline(
                "image-classification", model="cafeai/cafe_style"
            )
        elif name == "waifu":
            aesthetics["waifu"] = pipeline(
                "image-classification", model="cafeai/cafe_waifu"
            )


def judge_aesthetic(image):
    model_check("aesthetic")
    data = aesthetics["aesthetic"](image, top_k=2)
    result = {}
    for d in data:
        result[d["label"]] = d["score"]
    return result


def judge_style(image):
    model_check("style")
    data = aesthetics["style"](image, top_k=5)
    result = {}
    for d in data:
        result[d["label"]] = d["score"]
    return result


def judge_waifu(image):
    model_check("waifu")
    data = aesthetics["waifu"](image, top_k=5)
    result = {}
    for d in data:
        result[d["label"]] = d["score"]
    return result


def judge(image):
    if image is None:
        return None, None, None
    aesthetic = judge_aesthetic(image)
    style = judge_style(image)
    waifu = judge_waifu(image)
    return aesthetic, style, waifu

