# Merge block weighted Board
#
# extension of AUTOMATIC1111 web ui
#
# 2022/12/14 bbc_mc
#

import os
import gradio as gr

from modules import script_callbacks


from scripts.mbw import auto_mbw
from scripts.mbw_each import auto_mbw_mod

#
# UI callback
#
def on_ui_tabs():

    with gr.Blocks() as main_block:
        with gr.Tab("Auto MBW", elem_id="tab_auto_mbw"):
            auto_mbw.on_ui_tabs()
        with gr.Tab("Auto MBW Each", elem_id="tab_auto_mbw_each"):
            auto_mbw_mod.on_ui_tabs()

    # return required as (gradio_component, title, elem_id)
    return (main_block, "Auto MBW", "auto_mbw"),

# on_UI
script_callbacks.on_ui_tabs(on_ui_tabs)
