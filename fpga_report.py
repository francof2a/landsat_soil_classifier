#%%
import os
import json
from soil_classifier.utils import fpga_report

cwd = os.getcwd()
MODEL_NAME = 'ANN50x50th'
OUTPUT_FOLDER = cwd + '/outputs/'
HLS_PROJECT = 'hls_' + MODEL_NAME
FPGA_PROJECT = 'fpga_' + MODEL_NAME
REPORT_FILE = cwd + '/fpga/' + HLS_PROJECT + '/' + FPGA_PROJECT + '_prj/solution1/solution1_data.json'

#%% print
report = fpga_report(REPORT_FILE, FPGA_PROJECT)
print(report)

# %% save report
with open(OUTPUT_FOLDER+FPGA_PROJECT+'_report.json', 'w') as f:
    json.dump(report, f, indent=4)

# %%
