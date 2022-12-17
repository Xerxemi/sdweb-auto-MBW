# from https://note.com/kohya_ss/n/n9a485a066d5b
# kohya_ss
#   original code: https://github.com/eyriewow/merge-models

# use them as base of this code
# 2022/12/15
# bbc-mc

import os
import argparse
import re
import torch
from tqdm import tqdm

from modules import sd_models


NUM_INPUT_BLOCKS = 12
NUM_MID_BLOCK = 1
NUM_OUTPUT_BLOCKS = 12
NUM_TOTAL_BLOCKS = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + NUM_OUTPUT_BLOCKS


def dprint(str, flg):
    if flg:
        print(str)


def merge(weights:list, model_0, model_1, device="cpu", base_alpha=0.5, output_file="", allow_overwrite=False, verbose=False):
    if weights is None:
        weights = None
    else:
        weights = [float(w) for w in weights.split(',')]
    if len(weights) != NUM_TOTAL_BLOCKS:
        _err_msg = f"weights value must be {NUM_TOTAL_BLOCKS}."
        print(_err_msg)
        return False, _err_msg

    device = device if device in ["cpu", "cuda"] else "cpu"

    def load_model(_model, _device):
        model_info = sd_models.get_closet_checkpoint_match(_model)
        if model_info:
            model_file = model_info.filename
        return sd_models.read_state_dict(model_file, map_location=_device)

    print("loading", model_0)
    theta_0 = load_model(model_0, device)

    print("loading", model_1)
    theta_1 = load_model(model_1, device)

    alpha = base_alpha
    if not output_file or output_file == "":
        output_file = f'bw-{model_0}-{model_1}-{str(alpha)[2:] + "0"}.ckpt'
    else:
        output_file = output_file if ".ckpt" in output_file else output_file + ".ckpt"

    # check if output file already exists
    if os.path.isfile(output_file) and not allow_overwrite:
        _err_msg = f"Exiting... [{output_file}]"
        print(_err_msg)
        return False, _err_msg

    re_inp = re.compile(r'\.input_blocks\.(\d+)\.')  # 12
    re_mid = re.compile(r'\.middle_block\.(\d+)\.')  # 1
    re_out = re.compile(r'\.output_blocks\.(\d+)\.') # 12

    dprint(f"-- start Stage 1/2 --", verbose)
    count_target_of_basealpha = 0
    for key in (tqdm(theta_0.keys(), desc="Stage 1/2") if not verbose else theta_0.keys()):
        if "model" in key and key in theta_1:
            dprint(f"  key : {key}", verbose)
            current_alpha = alpha

            # check weighted and U-Net or not
            if weights is not None and 'model.diffusion_model.' in key:
                # check block index
                weight_index = -1

                if 'time_embed' in key:
                    weight_index = 0                # before input blocks
                elif '.out.' in key:
                    weight_index = NUM_TOTAL_BLOCKS - 1     # after output blocks
                else:
                    m = re_inp.search(key)
                    if m:
                        inp_idx = int(m.groups()[0])
                        weight_index = inp_idx
                    else:
                        m = re_mid.search(key)
                        if m:
                            weight_index = NUM_INPUT_BLOCKS
                        else:
                            m = re_out.search(key)
                            if m:
                                out_idx = int(m.groups()[0])
                                weight_index = NUM_INPUT_BLOCKS + NUM_MID_BLOCK + out_idx

                if weight_index >= NUM_TOTAL_BLOCKS:
                    print(f"error. illegal block index: {key}")
                    return False, ""
                if weight_index >= 0:
                    current_alpha = weights[weight_index]
                    dprint(f"weighted '{key}': {current_alpha}", verbose)
            else:
                count_target_of_basealpha = count_target_of_basealpha + 1
                dprint(f"base_alpha applied: [{key}]", verbose)

            theta_0[key] = (1 - current_alpha) * theta_0[key] + current_alpha * theta_1[key]

        else:
            dprint(f"  key - {key}", verbose)

    dprint(f"-- start Stage 2/2 --", verbose)
    for key in tqdm(theta_1.keys(), desc="Stage 2/2"):
        if "model" in key and key not in theta_0:
            dprint(f"  key : {key}", verbose)
            theta_0.update({key:theta_1[key]})
        else:
            dprint(f"  key - {key}", verbose)

    print("Saving...")

    torch.save({"state_dict": theta_0}, output_file)

    print("Done!")

    return True, f"{output_file}<br>base_alpha applied [{count_target_of_basealpha}] times."
