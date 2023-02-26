import datetime
from pathlib import Path

import gradio as gr
import os
import sys
import re
import statistics
import random

from modules import sd_models, shared, ui_components, processing, devices
from modules.ui_components import FormRow, FormGroup, ToolButton, FormHTML
from modules.shared import opts
import modules.shared as shared
from modules.sd_samplers import samplers
from tqdm import tqdm
try:
    from modules import hashes
    from modules.sd_models import CheckpointInfo
except:
    pass

from scripts.mbw.merge_block_weighted import merge
from scripts.util_autombw.txt2img_api import txt2img, refresh_models
from scripts.util_autombw.util_funcs import grouped
from scripts.mbw_util.merge_history import MergeHistory
from scripts.mbw_util.test_merge_history import TestMergeHistory
from scripts.mbw_util.preset_weights import PresetWeights

#classifier plugins
from importlib.machinery import SourceFileLoader

dirname = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
classifiers_path = os.path.join(dirname, "classifiers")

discovered_plugins = {}
plugins_count = 0
for _, dirs, _ in os.walk(classifiers_path):
    for directory in dirs:
        directory_path = os.path.join(classifiers_path, directory)
        # exclude __pycache__ directory
        if directory_path.endswith('__pycache__'):
            continue
        for module in os.listdir(directory_path):
            if module.startswith('score_') and module.endswith('.py'):
                module_name = os.path.splitext(module)[0]
                module_path = os.path.join(directory_path, module)
                discovered_plugins.update({module_name: SourceFileLoader(module_name, module_path).load_module()})
                plugins_count = plugins_count + 1

print("autoMBW: discovered " + str(plugins_count) + " classifier plugins.")

mergeHistory = MergeHistory()
testMergeHistory = TestMergeHistory()

presetWeights = PresetWeights()

def on_ui_tabs():
    def create_sampler_and_steps_selection(choices, tabname):
        with FormRow(elem_id=f"sampler_selection_{tabname}"):
            sampler = gr.Dropdown(label='Sampling method', elem_id=f"{tabname}_sampling", choices=[x.name for x in choices], value=choices[0].name)
            steps = gr.Slider(minimum=1, maximum=150, step=1, elem_id=f"{tabname}_steps", label="Sampling steps", value=20)
        return steps, sampler
    with gr.Column():
        with gr.Row():
            with gr.Column(variant="panel"):
                html_output_block_weight_info = gr.HTML()
                with gr.Row():
                    positive_prompt = gr.Text(label="Positive Prompt", elem_id="autombw_positive_prompt", lines=4, placeholder="Positive prompt here")
                    positive_prompt_2 = gr.Text(label="Positive Prompt", elem_id="autombw_positive_prompt_2", lines=4, placeholder="Positive prompt here")
                with gr.Row():
                    negative_prompt = gr.Text(label="Negative Prompt", elem_id="autombw_negative_prompt", lines=4, placeholder="Negative prompt here")
                    negative_prompt_2 = gr.Text(label="Negative Prompt", elem_id="autombw_negative_prompt_2", lines=4, placeholder="Negative prompt here")
                with gr.Row():
                    steps, sampler = create_sampler_and_steps_selection(samplers, "autombw")
                with FormRow():
                    with gr.Column(elem_id="autombw_column_size", scale=4):
                        width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512, elem_id="autombw_width")
                        height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512, elem_id="autombw_height")
                    with gr.Column(elem_id="autombw_column_batch"):
                        batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1, elem_id="autombw_batch_count")
                        batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1, elem_id="autombw_batch_size")
                cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0, elem_id="autombw_cfg_scale")
                with gr.Row():
                    chk_keep_random_seed = gr.Checkbox(label="Keep random seed", value=False)
                    chk_multi_pass_seed_progessive = gr.Checkbox(label="Multi pass progessive", value=False)
                    chk_multi_merge_seed_progessive = gr.Checkbox(label="Multi merge progessive", value=False)
                with FormRow(elem_id="autombw_checkboxes"):
                    restore_faces = gr.Checkbox(label='Restore faces', value=False, visible=len(shared.face_restorers) > 1, elem_id="autombw_restore_faces")
                    tiling = gr.Checkbox(label='Tiling', value=False, elem_id="autombw_tiling")
                    enable_hr = gr.Checkbox(label='Hires. fix', value=False, elem_id="autombw_enable_hr")
                    hr_final_resolution = FormHTML(value="", elem_id="txtimg_hr_finalres", label="Upscaled resolution", interactive=False)
                with FormGroup(visible=False, elem_id="txt2img_hires_fix") as hr_options:
                    with FormRow(elem_id="autombw_hires_fix_row1"):
                        hr_upscaler = gr.Dropdown(label="Upscaler", elem_id="autombw_hr_upscaler", choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]], value=shared.latent_upscale_default_mode)
                        hr_second_pass_steps = gr.Slider(minimum=0, maximum=150, step=1, label='Hires steps', value=0, elem_id="autombw_hires_steps")
                        denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.7, elem_id="autombw_denoising_strength")
                    with FormRow(elem_id="autombw_hires_fix_row2"):
                        hr_scale = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="Upscale by", value=2.0, elem_id="autombw_hr_scale")
                        hr_resize_x = gr.Slider(minimum=0, maximum=2048, step=64, label="Resize width to", value=0, elem_id="autombw_hr_resize_x")
                        hr_resize_y = gr.Slider(minimum=0, maximum=2048, step=64, label="Resize height to", value=0, elem_id="autombw_hr_resize_y")
                with gr.Row():
                        btn_do_merge_block_weighted = gr.Button(value="Run Merge", variant="primary", elem_id="autombw_do_merge_block_weighed")
                        btn_reload_checkpoint_mbw = gr.Button(value="Reload checkpoint", elem_id="autombw_reload_checkpoint")
            with gr.Column():
                txt_block_multi_merge = gr.Text(label="Multi Merge CMD", lines=4, elem_id="autombw_multi_merge")
                dd_preset_weight = gr.Dropdown(label="Preset Weights", choices=presetWeights.get_preset_name_list())
                txt_block_weight = gr.Text(label="Weight values", placeholder="Put weight sets. float number x 25")
                btn_apply_block_weight_from_txt = gr.Button(value="Apply block weight from text", variant="primary")
                with gr.Row():
                    with gr.Column(scale=50):
                        txt_block_test_increments = gr.Text(label="Test Increments", placeholder="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0", elem_id="autombw_test_increments")
                    with gr.Column(min_width=150):
                        chk_score_default = gr.Checkbox(label="Score default (nonlinear)", value=False, elem_id="autombw_score_default")
                with gr.Row():
                    with gr.Column():
                        sl_B_ALL = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Test Base", value=0, elem_id="autombw_test_base")
                        with gr.Row():
                            chk_base_alpha = gr.Checkbox(label="base_alpha", value=True, elem_id="autombw_base_alpha")
                            sl_base_alpha = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="base_alpha", value=0, elem_id="autombw_base_alpha_sl")
                    chk_verbose_mbw = gr.Checkbox(label="verbose console", value=False, elem_id="autombw_verbose_mbw")
                    chk_allow_overwrite = gr.Checkbox(label="Allow overwrite", value=True, interactive=False, elem_id="autombw_allow_overwrite")
                    chk_use_ramdisk = gr.Checkbox(label="Use ramdisk (Linux)", value=False, elem_id="autombw_use_ramdisk")
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            chk_save_as_half = gr.Checkbox(label="Save as half", value=True, elem_id="autombw_save_as_half")
                            chk_save_as_safetensors = gr.Checkbox(label="Save as safetensors", value=False, elem_id="autombw_save_as_safetensors")
                            chk_save_output_images = gr.Checkbox(label="Save output images", value=False, elem_id="autombw_save_output_images")
                    with gr.Column():
                        radio_position_ids = gr.Radio(label="Skip/Reset CLIP position_ids", choices=["None", "Skip", "Force Reset"], value="None", type="index", elem_id="autombw_position_ids")
                    with gr.Column():
                        with gr.Row():
                            dropdown_search_type = gr.Dropdown(label="Search Type", choices=["Linear", "Binary Mid Pass", "Binary", "Linear 2x", "Binary Mid Pass 2x", "Binary 2x"], value="Binary Mid Pass 2x", elem_id="autombw_search_type")
                        with gr.Row():
                            dropdown_classifiers = gr.Dropdown(label='Classifier', elem_id="autombw_classifiers", choices=[*discovered_plugins.keys()], value=[*discovered_plugins.keys()][0])
                    with gr.Column():
                        with gr.Row():
                            tally_types = ["Arithmetic Mean", "Geometric Mean", "Harmonic Mean", "A/G Mean", "G/H Mean", "A/H Mean",  "Median", "Min", "Max", "Min*Max", "Fuzz Mode"]
                            dropdown_tally_type = gr.Dropdown(label="Tally Type", choices=tally_types, value="Arithmetic Mean", elem_id="autombw_tally_type")
                            dropdown_tally_type_2 = gr.Dropdown(label="Tally Type 2", choices=tally_types, value="Arithmetic Mean", elem_id="autombw_tally_type_2")
                            dropdown_tally_type_3 = gr.Dropdown(label="Tally Type 3", choices=tally_types, value="Arithmetic Mean", elem_id="autombw_tally_type_3")
                        with gr.Row():
                            dropdown_pass_count = gr.Dropdown(label="Passes", choices=['Singlepass', 'Doublepass', 'Triplepass'], value='Singlepass', elem_id="autombw_pass_count", type="index")
        with gr.Row():
            model_A = gr.Dropdown(label="Model A", choices=sd_models.checkpoint_tiles(), elem_id="autombw_model_a")
            model_B = gr.Dropdown(label="Model B", choices=sd_models.checkpoint_tiles(), elem_id="autombw_model_b")
            txt_model_O = gr.Text(label="Output Model Name", elem_id="autombw_model_o")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    chk_IN_00 = gr.Checkbox(label="IN00", value=True, elem_id="autombw_in00")
                    sl_IN_00 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="IN00", value=0, elem_id="autombw_in00_sl")
                with gr.Row():
                    chk_IN_01 = gr.Checkbox(label="IN01", value=True, elem_id="autombw_in01")
                    sl_IN_01 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="IN01", value=0, elem_id="autombw_in01_sl")
                with gr.Row():
                    chk_IN_02 = gr.Checkbox(label="IN02", value=True, elem_id="autombw_in02")
                    sl_IN_02 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="IN02", value=0, elem_id="autombw_in02_sl")
                with gr.Row():
                    chk_IN_03 = gr.Checkbox(label="IN03", value=True, elem_id="autombw_in03")
                    sl_IN_03 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="IN03", value=0, elem_id="autombw_in03_sl")
                with gr.Row():
                    chk_IN_04 = gr.Checkbox(label="IN04", value=True, elem_id="autombw_in04")
                    sl_IN_04 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="IN04", value=0, elem_id="autombw_in04_sl")
                with gr.Row():
                    chk_IN_05 = gr.Checkbox(label="IN05", value=True, elem_id="autombw_in05")
                    sl_IN_05 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="IN05", value=0, elem_id="autombw_in05_sl")
                with gr.Row():
                    chk_IN_06 = gr.Checkbox(label="IN06", value=True, elem_id="autombw_in06")
                    sl_IN_06 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="IN06", value=0, elem_id="autombw_in06_sl")
                with gr.Row():
                    chk_IN_07 = gr.Checkbox(label="IN07", value=True, elem_id="autombw_in07")
                    sl_IN_07 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="IN07", value=0, elem_id="autombw_in07_sl")
                with gr.Row():
                    chk_IN_08 = gr.Checkbox(label="IN08", value=True, elem_id="autombw_in08")
                    sl_IN_08 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="IN08", value=0, elem_id="autombw_in08_sl")
                with gr.Row():
                    chk_IN_09 = gr.Checkbox(label="IN09", value=True, elem_id="autombw_in09")
                    sl_IN_09 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="IN09", value=0, elem_id="autombw_in09_sl")
                with gr.Row():
                    chk_IN_10 = gr.Checkbox(label="IN10", value=True, elem_id="autombw_in10")
                    sl_IN_10 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="IN10", value=0, elem_id="autombw_in10_sl")
                with gr.Row():
                    chk_IN_11 = gr.Checkbox(label="IN11", value=True, elem_id="autombw_in11")
                    sl_IN_11 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="IN11", value=0, elem_id="autombw_in11_sl")
            with gr.Column():
                with gr.Row():
                    chk_M_00 = gr.Checkbox(label="M00", value=True, elem_id="autombw_m00")
                    sl_M_00 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="M00", value=0, elem_id="autombw_m00_sl")
            with gr.Column():
                with gr.Row():
                    chk_OUT_00 = gr.Checkbox(label="OUT00", value=True, elem_id="autombw_out00")
                    sl_OUT_00 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="OUT00", value=0, elem_id="autombw_out00_sl")
                with gr.Row():
                    chk_OUT_01 = gr.Checkbox(label="OUT01", value=True, elem_id="autombw_out01")
                    sl_OUT_01 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="OUT01", value=0, elem_id="autombw_out01_sl")
                with gr.Row():
                    chk_OUT_02 = gr.Checkbox(label="OUT02", value=True, elem_id="autombw_out02")
                    sl_OUT_02 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="OUT02", value=0, elem_id="autombw_out02_sl")
                with gr.Row():
                    chk_OUT_03 = gr.Checkbox(label="OUT03", value=True, elem_id="autombw_out03")
                    sl_OUT_03 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="OUT03", value=0, elem_id="autombw_out03_sl")
                with gr.Row():
                    chk_OUT_04 = gr.Checkbox(label="OUT04", value=True, elem_id="autombw_out04")
                    sl_OUT_04 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="OUT04", value=0, elem_id="autombw_out04_sl")
                with gr.Row():
                    chk_OUT_05 = gr.Checkbox(label="OUT05", value=True, elem_id="autombw_out05")
                    sl_OUT_05 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="OUT05", value=0, elem_id="autombw_out05_sl")
                with gr.Row():
                    chk_OUT_06 = gr.Checkbox(label="OUT06", value=True, elem_id="autombw_out06")
                    sl_OUT_06 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="OUT06", value=0, elem_id="autombw_out06_sl")
                with gr.Row():
                    chk_OUT_07 = gr.Checkbox(label="OUT07", value=True, elem_id="autombw_out07")
                    sl_OUT_07 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="OUT07", value=0, elem_id="autombw_out07_sl")
                with gr.Row():
                    chk_OUT_08 = gr.Checkbox(label="OUT08", value=True, elem_id="autombw_out08")
                    sl_OUT_08 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="OUT08", value=0, elem_id="autombw_out08_sl")
                with gr.Row():
                    chk_OUT_09 = gr.Checkbox(label="OUT09", value=True, elem_id="autombw_out09")
                    sl_OUT_09 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="OUT09", value=0, elem_id="autombw_out09_sl")
                with gr.Row():
                    chk_OUT_10 = gr.Checkbox(label="OUT10", value=True, elem_id="autombw_out10")
                    sl_OUT_10 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="OUT10", value=0, elem_id="autombw_out10_sl")
                with gr.Row():
                    chk_OUT_11 = gr.Checkbox(label="OUT11", value=True, elem_id="autombw_out11")
                    sl_OUT_11 = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="OUT11", value=0, elem_id="autombw_out11_sl")

    chks = [
        chk_IN_00, chk_IN_01, chk_IN_02, chk_IN_03, chk_IN_04, chk_IN_05,
        chk_IN_06, chk_IN_07, chk_IN_08, chk_IN_09, chk_IN_10, chk_IN_11,
        chk_M_00,
        chk_OUT_00, chk_OUT_01, chk_OUT_02, chk_OUT_03, chk_OUT_04, chk_OUT_05,
        chk_OUT_06, chk_OUT_07, chk_OUT_08, chk_OUT_09, chk_OUT_10, chk_OUT_11,
        chk_base_alpha]

    sliders = [sl_IN_00, sl_IN_01, sl_IN_02, sl_IN_03, sl_IN_04, sl_IN_05,
        sl_IN_06, sl_IN_07, sl_IN_08, sl_IN_09, sl_IN_10, sl_IN_11,
        sl_M_00,
        sl_OUT_00, sl_OUT_01, sl_OUT_02, sl_OUT_03, sl_OUT_04, sl_OUT_05,
        sl_OUT_06, sl_OUT_07, sl_OUT_08, sl_OUT_09, sl_OUT_10, sl_OUT_11,
        sl_base_alpha]

    # Events
    def onclick_btn_do_merge_block_weighted(
        model_A, model_B,
        chk_IN_00, chk_IN_01, chk_IN_02, chk_IN_03, chk_IN_04, chk_IN_05,
        chk_IN_06, chk_IN_07, chk_IN_08, chk_IN_09, chk_IN_10, chk_IN_11,
        chk_M_00,
        chk_OUT_00, chk_OUT_01, chk_OUT_02, chk_OUT_03, chk_OUT_04, chk_OUT_05,
        chk_OUT_06, chk_OUT_07, chk_OUT_08, chk_OUT_09, chk_OUT_10, chk_OUT_11,
        chk_base_alpha,
        sl_IN_00, sl_IN_01, sl_IN_02, sl_IN_03, sl_IN_04, sl_IN_05,
        sl_IN_06, sl_IN_07, sl_IN_08, sl_IN_09, sl_IN_10, sl_IN_11,
        sl_M_00,
        sl_OUT_00, sl_OUT_01, sl_OUT_02, sl_OUT_03, sl_OUT_04, sl_OUT_05,
        sl_OUT_06, sl_OUT_07, sl_OUT_08, sl_OUT_09, sl_OUT_10, sl_OUT_11,
        sl_base_alpha,
        txt_model_O, chk_verbose_mbw, chk_allow_overwrite, chk_use_ramdisk,
        chk_save_as_safetensors, chk_save_as_half, chk_save_output_images,
        radio_position_ids,
        dropdown_search_type, dropdown_classifiers, dropdown_pass_count, dropdown_tally_type, dropdown_tally_type_2, dropdown_tally_type_3,
        txt_block_test_increments, chk_score_default, txt_block_multi_merge,
        enable_hr, denoising_strength, hr_scale, hr_upscaler, hr_second_pass_steps, hr_resize_x, hr_resize_y,
        positive_prompt, positive_prompt_2, chk_keep_random_seed, chk_multi_pass_seed_progessive, chk_multi_merge_seed_progessive, sampler, batch_size, batch_count, steps, cfg_scale, width, height, restore_faces, tiling, negative_prompt, negative_prompt_2
    ):
        print( "#### Merge Block Weighted ####")

        #parsing multi merge txt block
        multi_model_A = []
        multi_model_B = []
        multi_model_O = []
        disable_singular_merge = False
        if txt_block_multi_merge.strip() != "":
            for line in txt_block_multi_merge.splitlines():
                multi_model_A.append(line.split("+")[0].strip())
                multi_model_B.append(line.split("+")[1].split("=")[0].strip())
                multi_model_O.append(line.split("+")[1].split("=")[1].strip())
            if len(multi_model_A) == len(multi_model_B) == len(multi_model_O):
                disable_singular_merge = True
                print("autoMBW: multi merge detected.")
            else:
                print("autoMBW: multi merge parse error.")
        if not disable_singular_merge:
            multi_model_A = [model_A]
            multi_model_B = [model_B]
            multi_model_O = [txt_model_O]

        #weird webui replacement shenanigans
        random_number = 0

        #seed settings
        if chk_keep_random_seed == True:
            seed = -1
        else:
            seed = 1

        prechks = {"25":chk_base_alpha, "12":chk_M_00}

        chks = {
        "0":chk_IN_00, "1":chk_IN_01, "2":chk_IN_02, "3":chk_IN_03, "4":chk_IN_04, "5":chk_IN_05,
        "6":chk_IN_06, "7":chk_IN_07, "8":chk_IN_08, "9":chk_IN_09, "10":chk_IN_10, "11":chk_IN_11,
        "13":chk_OUT_00, "14":chk_OUT_01, "15":chk_OUT_02, "16":chk_OUT_03, "17":chk_OUT_04, "18":chk_OUT_05,
        "19":chk_OUT_06, "20":chk_OUT_07, "21":chk_OUT_08, "22":chk_OUT_09, "23":chk_OUT_10, "24":chk_OUT_11}

        sliders = {
        "0":sl_IN_00, "1":sl_IN_01, "2":sl_IN_02, "3":sl_IN_03, "4":sl_IN_04, "5":sl_IN_05,
        "6":sl_IN_06, "7":sl_IN_07, "8":sl_IN_08, "9":sl_IN_09, "10":sl_IN_10, "11":sl_IN_11,
        "12":sl_M_00,
        "13":sl_OUT_00, "14":sl_OUT_01, "15":sl_OUT_02, "16":sl_OUT_03, "17":sl_OUT_04, "18":sl_OUT_05,
        "19":sl_OUT_06, "20":sl_OUT_07, "21":sl_OUT_08, "22":sl_OUT_09, "23":sl_OUT_10, "24":sl_OUT_11,
        "25":sl_base_alpha}

        #parsing test increments
        try:
            test_increments = [float(n) for n in str(txt_block_test_increments).split(',')].sort()
            if len(test_increments) < 2:
                raise ValueError()
            for num in test_increments:
                if num < 0 or num > 1:
                    raise ValueError("testvals must be within 0 and 1")
        except:
            test_increments = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

        def merge_wrapper(keyval, final=False):
            print( "starting merge...")

            base_alpha = savedweights[25]
            testweights = savedweights[:-1]

            if not final:
                for key in keyval:
                    if int(key) == 25:
                        base_alpha = keyval[key]
                    else:
                        testweights[int(key)] = keyval[key]
            _weights = ','.join([str(i) for i in testweights])

            if not model_A or not model_B:
                gr.update(value=f"ERROR: model not found. [{model_A}][{model_B}]")

            #
            # Prepare params before run merge
            #

            # generate output file name from param
            model_A_info = sd_models.get_closet_checkpoint_match(model_A)
            if model_A_info:
                _model_A_name = model_A_info.model_name
            else:
                _model_A_name = ""
            model_B_info = sd_models.get_closet_checkpoint_match(model_B)
            if model_B_info:
                _model_B_info = model_B_info.model_name
            else:
                _model_B_info = ""

            def validate_output_filename(output_filename, save_as_safetensors=False, save_as_half=False):
                output_filename = re.sub(r'[\\|:|?|"|<|>|\|\*]', '-', output_filename)
                filename_body, filename_ext = os.path.splitext(output_filename)
                _ret = output_filename
                _footer = "-half" if save_as_half else ""

                #to get around webui not handling replacements correctly
                if not final:
                    nonlocal random_number
                    while True:
                        local_random_number = random.randint(0, 1000)
                        if local_random_number != random_number:
                            random_number = local_random_number
                            break
                    _footer = _footer + str(random_number)

                if filename_ext in [".safetensors", ".ckpt"]:
                    _ret = f"{filename_body}{_footer}{filename_ext}"
                elif save_as_safetensors:
                    _ret = f"{output_filename}{_footer}.safetensors"
                else:
                    _ret = f"{output_filename}{_footer}.ckpt"
                return _ret

            model_O = f"bw-merge-{_model_A_name}-{_model_B_info}-{base_alpha}.ckpt" if txt_model_O == "" else txt_model_O
            model_O = validate_output_filename(model_O, save_as_safetensors=chk_save_as_safetensors, save_as_half=chk_save_as_half)

            _output = os.path.join(shared.cmd_opts.ckpt_dir or sd_models.model_path, model_O)

            # if not chk_allow_overwrite:
            #     if os.path.exists(_output):
            #         _err_msg = f"ERROR: output_file already exists. overwrite not allowed. abort."
            #         print(_err_msg)
            #         return gr.update(value=f"{_err_msg} [{_output}]")

            print(f"  model_0    : {model_A}")
            print(f"  model_1    : {model_B}")
            print(f"  base_alpha : {base_alpha}")
            print(f"  output_file: {_output}")
            print(f"  weights    : {_weights}")
            print(f"  skip ids   : {radio_position_ids} : 0:None, 1:Skip, 2:Reset")

            if final:
                print(f"  test type  : Final")
                local_chk_use_ramdisk = False
            else:
                print(f"  test type  : Test")
                local_chk_use_ramdisk = chk_use_ramdisk

            result, ret_message = merge(weights=_weights, model_0=model_A, model_1=model_B, allow_overwrite=chk_allow_overwrite,
                base_alpha=base_alpha, output_file=_output, verbose=chk_verbose_mbw,
                save_as_safetensors=chk_save_as_safetensors,
                save_as_half=chk_save_as_half,
                skip_position_ids=radio_position_ids,
                use_ramdisk=local_chk_use_ramdisk
                )

            if result:
                ret_html = "merged.<br>" \
                    + f"{model_A}<br>" \
                    + f"{model_B}<br>" \
                    + f"{model_O}<br>" \
                    + f"base_alpha={base_alpha}<br>" \
                    + f"Weight_values={_weights}<br>"
                print("merged.")
                refresh_models()
                print("starting generation.")
                images = txt2img(model=model_O,
                    enable_hr=enable_hr, denoising_strength=denoising_strength, firstphase_width=0, firstphase_height=0, hr_scale=hr_scale, hr_upscaler=hr_upscaler, hr_second_pass_steps=hr_second_pass_steps, hr_resize_x=hr_resize_x, hr_resize_y=hr_resize_y,
                    prompt=positive_prompt, seed=seed, sampler_name=sampler, batch_size=batch_size, n_iter=batch_count, steps=steps, cfg_scale=cfg_scale, width=width, height=height, restore_faces=restore_faces, tiling=tiling, negative_prompt=negative_prompt)
                if positive_prompt_2 != "" or negative_prompt_2 != "":
                    images2 = txt2img(model=model_O,
                        enable_hr=enable_hr, denoising_strength=denoising_strength, firstphase_width=0, firstphase_height=0, hr_scale=hr_scale, hr_upscaler=hr_upscaler, hr_second_pass_steps=hr_second_pass_steps, hr_resize_x=hr_resize_x, hr_resize_y=hr_resize_y,
                        prompt=positive_prompt_2, seed=seed, sampler_name=sampler, batch_size=batch_size, n_iter=batch_count, steps=steps, cfg_scale=cfg_scale, width=width, height=height, restore_faces=restore_faces, tiling=tiling, negative_prompt=negative_prompt_2)
                else:
                    images2 = []
                # timestamp without spaces
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                folder_name = f"{_model_A_name}-{_model_B_info}-{timestamp}"
                folder_path = os.path.join(shared.cmd_opts.data_dir, 'auto_mbw_output', folder_name)
                os.makedirs(folder_path)
                with open(os.path.join(folder_path, "000-weights.txt"), 'w') as f:
                    f.write(_weights)
                    f.write('\n')
                    f.write("Base Alpha:")
                    f.write(str(base_alpha))
                imagescores = []
                for idx, image in enumerate(images + images2):
                    #classifier plugin stuff
                    classifier = discovered_plugins[dropdown_classifiers]
                    score = classifier.score(image)
                    imagescores.append(score)
                    if chk_save_output_images:
                        output_name = f"{idx}.png"
                        image.save(os.path.join(folder_path, output_name))

                normscores = [float(i)/max(imagescores) for i in imagescores]

                if current_pass == 0:
                    local_dropdown_tally_type = dropdown_tally_type
                elif current_pass == 1:
                    local_dropdown_tally_type = dropdown_tally_type_2
                elif current_pass == 2:
                    local_dropdown_tally_type = dropdown_tally_type_3

                if local_dropdown_tally_type == "Arithmetic Mean":
                    testscore = statistics.mean(imagescores)
                elif local_dropdown_tally_type == "Geometric Mean":
                    testscore = statistics.geometric_mean(imagescores)
                elif local_dropdown_tally_type == "Harmonic Mean":
                    testscore = statistics.harmonic_mean(imagescores)
                elif local_dropdown_tally_type == "A/G Mean":
                    testscore = (statistics.mean(normscores)/statistics.geometric_mean(normscores))*statistics.mean(imagescores)
                elif local_dropdown_tally_type == "G/H Mean":
                    testscore = (statistics.geometric_mean(normscores)/statistics.harmonic_mean(normscores))*statistics.mean(imagescores)
                elif local_dropdown_tally_type == "A/H Mean":
                    testscore = (statistics.mean(normscores)/statistics.harmonic_mean(normscores))*statistics.mean(imagescores)
                elif local_dropdown_tally_type == "Median":
                    testscore = statistics.median(imagescores)
                elif local_dropdown_tally_type == "Min":
                    testscore = min(imagescores)
                elif local_dropdown_tally_type == "Max":
                    testscore = max(imagescores)
                elif local_dropdown_tally_type == "Min*Max":
                    testscore = (min(imagescores)*max(imagescores)) ** 0.5
                elif local_dropdown_tally_type == "Fuzz Mode":
                    fuzzyscores = []
                    for score in normscores:
                        for _ in range(1000):
                            fuzzyscores.append(round(score+(random.random()/100), 2))
                    testscore = statistics.mode(fuzzyscores)*max(imagescores)
                if not final:
                    if os.path.islink(_output):
                        os.remove(os.path.realpath(_output))
                    os.remove(_output)
                print("test score: " + str(testscore))
                refresh_models()

            else:
                ret_html = ret_message
                print("merge failed.")

            # save log to history.tsv
            weight_name = ""
            if final:
                sd_models.list_models()
                model_A_info = sd_models.get_closet_checkpoint_match(model_A)
                model_B_info = sd_models.get_closet_checkpoint_match(model_B)
                model_O_info = sd_models.get_closet_checkpoint_match(os.path.basename(_output))
                if hasattr(model_O_info, "sha256") and model_O_info.sha256 is None:
                    model_O_info:CheckpointInfo = model_O_info
                    model_O_info.sha256 = hashes.sha256(model_O_info.filename, "checkpoint/" + model_O_info.title)
                def model_name(model_info):
                    return model_info.name if hasattr(model_info, "name") else model_info.title
                def model_sha256(model_info):
                    return model_info.sha256 if hasattr(model_info, "sha256") else ""
                mergeHistory.add_history(
                        model_name(model_A_info),
                        model_A_info.hash,
                        model_sha256(model_A_info),
                        model_name(model_B_info),
                        model_B_info.hash,
                        model_sha256(model_B_info),
                        model_name(model_O_info),
                        model_O_info.hash,
                        model_sha256(model_O_info),
                        base_alpha,
                        _weights,
                        "",
                        weight_name,
                        positive_prompt,
                        negative_prompt
                        )
                return ret_html
            else:
                testMergeHistory.add_history(
                    base_alpha,
                    _weights,
                    "",
                    weight_name,
                    positive_prompt,
                    negative_prompt
                    )
                return testscore

        def arrange_keys(keys, testval):
            merge_params = {}
            for key in keys:
                merge_params.update({key: testval})
            return merge_params

        def linear_pass(chks, window):
            nonlocal savedweights
            nonlocal scores
            for keys in grouped(chks, window):
                if all(bool(chks[key]) for key in keys):
                    testscores = {}
                    for testval in test_increments:
                        merge_params = arrange_keys(keys, testval)
                        testscores.update({str(testval):merge_wrapper(merge_params)})
                    for key in keys:
                        savedweights[int(key)] = float(max(testscores, key=testscores.get))
                        scores.update({str(key):testscores})

        def binary_pass(chks, window):
            nonlocal savedweights
            nonlocal scores
            full_keys = [*chks.keys()]
            next_carry = (None, 0)
            for keys in grouped(chks, window):
                if all(bool(chks[key]) for key in keys):
                    binary_test_increments = test_increments
                    lowerscore = 1
                    upperscore = 1
                    carry_val = None
                    while True:
                        lower = binary_test_increments[0]
                        upper = binary_test_increments[-1]
                        if len(binary_test_increments) == 1:
                            if chk_score_default == False or next_carry[1] < carry_val or not carry_val:
                                for key in keys:
                                    savedweights[int(key)] = float(lower)
                                    scores.update({str(key):{str(lower):lowerscore}})
                            else:
                                carry_val = next_carry[1]
                            if int(full_keys.index(keys[-1])) + len(keys) < len(full_keys):
                                next_key = int(full_keys[int(full_keys.index(keys[-1])) + len(keys)])
                                if len(savedweights) > next_key:
                                    next_carry = (savedweights[next_key], carry_val)
                            break

                        if lower == next_carry[0]:
                            lowerscore = next_carry[1]
                        else:
                            if carry_val != lowerscore:
                                merge_params = arrange_keys(keys, lower)
                                lowerscore = merge_wrapper(merge_params)
                        if upper == next_carry[0]:
                            upperscore = next_carry[1]
                        else:
                            if carry_val != upperscore:
                                merge_params = arrange_keys(keys, upper)
                                upperscore = merge_wrapper(merge_params)

                        if lowerscore > upperscore:
                            binary_test_increments = binary_test_increments[:len(binary_test_increments)//2]
                            carry_val = lowerscore
                        elif upperscore > lowerscore:
                            binary_test_increments = binary_test_increments[len(binary_test_increments)//2:]
                            carry_val = upperscore

        def binary_mid_pass(chks, window):
            nonlocal savedweights
            nonlocal scores
            full_keys = [*chks.keys()]
            next_carry = (None, 0)
            for keys in grouped(chks, window):
                if all(bool(chks[key]) for key in keys):
                    binary_test_increments = test_increments
                    lowerscore = 1
                    middlescore = 1
                    upperscore = 1
                    carry_dict = {}
                    if next_carry[0] != None:
                        carry_dict.update({next_carry[0]:next_carry[1]})
                    while True:
                        lower = binary_test_increments[0]
                        middle = binary_test_increments[((len(binary_test_increments)+1)//2)-1]
                        upper = binary_test_increments[-1]
                        if len(binary_test_increments) == 1:
                            if chk_score_default == False or next_carry[1] < max([*carry_dict.values()]):
                                for key in keys:
                                    savedweights[int(key)] = float(lower)
                                    scores.update({str(key):{str(lower):lowerscore}})
                            if int(full_keys.index(keys[-1])) + len(keys) < len(full_keys):
                                next_key = int(full_keys[int(full_keys.index(keys[-1])) + len(keys)])
                                if len(savedweights) > next_key:
                                    next_carry = (savedweights[next_key], max([*carry_dict.values()]))
                            break

                        if lower in [*carry_dict.keys()]:
                            lowerscore = carry_dict[lower]
                        else:
                            merge_params = arrange_keys(keys, lower)
                            lowerscore = merge_wrapper(merge_params)
                        if middle in [*carry_dict.keys()]:
                            middlescore = carry_dict[middle]
                        else:
                            merge_params = arrange_keys(keys, middle)
                            middlescore = merge_wrapper(merge_params)
                        if upper in [*carry_dict.keys()]:
                            upperscore = carry_dict[upper]
                        else:
                            merge_params = arrange_keys(keys, upper)
                            upperscore = merge_wrapper(merge_params)

                        score_list = [lowerscore, middlescore, upperscore]
                        carry_dict.update({lower:lowerscore, middle:middlescore, upper:upperscore})
                        if lowerscore == max(score_list):
                            binary_test_increments = binary_test_increments[:len(binary_test_increments)//2]
                        elif middlescore == max(score_list):
                            binary_test_increments = binary_test_increments[len(binary_test_increments)//4:-len(binary_test_increments)//4]
                        elif upperscore == max(score_list):
                            binary_test_increments = binary_test_increments[len(binary_test_increments)//2:]

        for model_A, model_B, txt_model_O in zip(multi_model_A, multi_model_B, multi_model_O):
            if chk_multi_merge_seed_progessive == True and seed != -1:
                seed = seed + 1
            savedweights = [float(n) for n in [*sliders.values()]]
            scores = {}
            for current_pass in range(dropdown_pass_count + 1):
                if chk_multi_pass_seed_progessive == True and seed != -1:
                    seed = seed + 1
                #prechks - base alpha & m00
                if dropdown_search_type in ["Linear", "Linear 2x"]:
                    linear_pass(prechks, 1)
                if dropdown_search_type in ["Binary Mid Pass", "Binary Mid Pass 2x"]:
                    binary_mid_pass(prechks, 1)
                elif dropdown_search_type in ["Binary", "Binary 2x"]:
                    binary_pass(prechks, 1)
                #chks - IN&OUT layers
                if dropdown_search_type == "Linear":
                    linear_pass(chks, 1)
                elif dropdown_search_type == "Binary Mid Pass":
                    binary_mid_pass(chks, 1)
                elif dropdown_search_type == "Binary":
                    binary_pass(chks, 1)
                elif dropdown_search_type == "Linear 2x":
                    linear_pass(chks, 2)
                elif dropdown_search_type == "Binary Mid Pass 2x":
                    binary_mid_pass(chks, 2)
                elif dropdown_search_type == "Binary 2x":
                    binary_pass(chks, 2)
            ret_html = merge_wrapper({"25": savedweights[25]}, True)

        return gr.update(value=f"{ret_html}")

    btn_do_merge_block_weighted.click(
        fn=onclick_btn_do_merge_block_weighted,
        inputs=[model_A, model_B]
            + chks
            + sliders
            + [txt_model_O, chk_verbose_mbw, chk_allow_overwrite, chk_use_ramdisk]
            + [chk_save_as_safetensors, chk_save_as_half, chk_save_output_images, radio_position_ids]
            + [dropdown_search_type, dropdown_classifiers, dropdown_pass_count, dropdown_tally_type, dropdown_tally_type_2, dropdown_tally_type_3]
            + [txt_block_test_increments, chk_score_default, txt_block_multi_merge]
            + [enable_hr, denoising_strength, hr_scale, hr_upscaler, hr_second_pass_steps, hr_resize_x, hr_resize_y]
            + [positive_prompt, positive_prompt_2, chk_keep_random_seed, chk_multi_pass_seed_progessive, chk_multi_merge_seed_progessive, sampler, batch_size, batch_count, steps, cfg_scale, width, height, restore_faces, tiling, negative_prompt, negative_prompt_2],
        outputs=[html_output_block_weight_info]
    )

    def on_btn_reload_checkpoint_mbw():
        sd_models.list_models()
        return [gr.update(choices=sd_models.checkpoint_tiles()), gr.update(choices=sd_models.checkpoint_tiles())]
    btn_reload_checkpoint_mbw.click(
        fn=on_btn_reload_checkpoint_mbw,
        inputs=[],
        outputs=[model_A, model_B]
    )

    def calc_resolution_hires(enable, width, height, hr_scale, hr_resize_x, hr_resize_y):
        if not enable:
            return ""
        p = processing.StableDiffusionProcessingTxt2Img(width=width, height=height, enable_hr=True, hr_scale=hr_scale, hr_resize_x=hr_resize_x, hr_resize_y=hr_resize_y)
        with devices.autocast():
            p.init([""], [0], [0])
        return f"resize: from <span class='resolution'>{p.width}x{p.height}</span> to <span class='resolution'>{p.hr_resize_x or p.hr_upscale_to_x}x{p.hr_resize_y or p.hr_upscale_to_y}</span>"

    hr_resolution_preview_inputs = [enable_hr, width, height, hr_scale, hr_resize_x, hr_resize_y]
    for input in hr_resolution_preview_inputs:
        input.change(
            fn=calc_resolution_hires,
            inputs=hr_resolution_preview_inputs,
            outputs=[hr_final_resolution],
            show_progress=False,
        )
        input.change(
            None,
            _js="onCalcResolutionHires",
            inputs=hr_resolution_preview_inputs,
            outputs=[],
            show_progress=False,
        )

    def gr_show(visible=True):
        return {"visible": visible, "__type__": "update"}

    enable_hr.change(
    fn=lambda x: gr_show(x),
    inputs=[enable_hr],
    outputs=[hr_options],
    show_progress = False,
    )

    def on_change_test_base(test_base):
        _list = [test_base] * 26
        return [gr.update(value=x, visible=True) for x in _list]
    sl_B_ALL.change(
        fn=on_change_test_base,
        inputs=[sl_B_ALL],
        outputs=[
            sl_IN_00, sl_IN_01, sl_IN_02, sl_IN_03, sl_IN_04, sl_IN_05,
            sl_IN_06, sl_IN_07, sl_IN_08, sl_IN_09, sl_IN_10, sl_IN_11,
            sl_M_00,
            sl_OUT_00, sl_OUT_01, sl_OUT_02, sl_OUT_03, sl_OUT_04, sl_OUT_05,
            sl_OUT_06, sl_OUT_07, sl_OUT_08, sl_OUT_09, sl_OUT_10, sl_OUT_11,
            sl_base_alpha
        ]
    )

    def on_change_dd_preset_weight(dd_preset_weight):
        _weights = presetWeights.find_weight_by_name(dd_preset_weight)
        return _weights
    dd_preset_weight.change(
        fn=on_change_dd_preset_weight,
        inputs=[dd_preset_weight],
        outputs=[txt_block_weight]
    )

    def on_btn_apply_block_weight_from_txt(txt_block_weight):
        if not txt_block_weight or txt_block_weight == "":
            return [gr.update() for _ in range(25)]
        _list = [x.strip() for x in txt_block_weight.split(",")]
        if(len(_list) != 25):
            return [gr.update() for _ in range(25)]
        return [gr.update(value=x, visible=True) for x in _list]
    btn_apply_block_weight_from_txt.click(
        fn=on_btn_apply_block_weight_from_txt,
        inputs=[txt_block_weight],
        outputs=[
            sl_IN_00, sl_IN_01, sl_IN_02, sl_IN_03, sl_IN_04, sl_IN_05,
            sl_IN_06, sl_IN_07, sl_IN_08, sl_IN_09, sl_IN_10, sl_IN_11,
            sl_M_00,
            sl_OUT_00, sl_OUT_01, sl_OUT_02, sl_OUT_03, sl_OUT_04, sl_OUT_05,
            sl_OUT_06, sl_OUT_07, sl_OUT_08, sl_OUT_09, sl_OUT_10, sl_OUT_11,
        ]
    )

