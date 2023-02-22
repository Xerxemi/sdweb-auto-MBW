#
#
#
import os
import datetime
from csv import DictWriter, DictReader
import shutil

from modules import scripts


CSV_FILE_ROOT = "csv/"
CSV_FILE_PATH = "csv/test_history.tsv"
HEADERS = [
        "base_alpha", "weight_name", "weight_values", "weight_values2", "datetime", "positive_prompt", "negative_prompt"]
path_root = scripts.basedir()

class TestMergeHistory():
    def __init__(self):
        self.fileroot = os.path.join(path_root, CSV_FILE_ROOT)
        self.filepath = os.path.join(path_root, CSV_FILE_PATH)
        if not os.path.exists(self.fileroot):
            os.mkdir(self.fileroot)
        if os.path.exists(self.filepath):
            self.update_header()

    def add_history(self,
                sl_base_alpha,
                weight_value_A,
                weight_value_B,
                weight_name,
                positive_prompt, negative_prompt):
        _history_dict = {}
        _history_dict.update({
            "base_alpha": sl_base_alpha,
            "weight_name": weight_name,
            "weight_values": weight_value_A,
            "weight_values2": weight_value_B,
            "datetime": f"{datetime.datetime.now()}",
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt
            })

        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                dw = DictWriter(f, fieldnames=HEADERS, delimiter='\t')
                dw.writeheader()
        # save to file
        with open(self.filepath, "a", newline="", encoding='utf-8') as f:
            dw = DictWriter(f, fieldnames=HEADERS, delimiter='\t')
            dw.writerow(_history_dict)

    def update_header(self):
        hist_data = []
        if os.path.exists(self.filepath):
            # check header in case HEADERS updated
            with open(self.filepath, "r", newline="", encoding="utf-8") as f:
                dr = DictReader(f, delimiter='\t')
                new_header = [ x for x in HEADERS if x not in dr.fieldnames ]
                if len(new_header) > 0:
                    # need update.
                    hist_data = [ x for x in dr]
            # apply change
            if len(hist_data) > 0:
                # backup before change
                shutil.copy(self.filepath, self.filepath + ".bak")
                with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                    dw = DictWriter(f, fieldnames=HEADERS, delimiter='\t')
                    dw.writeheader()
                    dw.writerows(hist_data)
