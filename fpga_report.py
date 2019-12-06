#%%
import os
import json

cwd = os.getcwd()
MODEL_NAME = 'ANN50x50th'
OUTPUT_FOLDER = cwd + '/outputs/'
HLS_PROJECT = 'hls_' + MODEL_NAME
FPGA_PROJECT = 'fpga_' + MODEL_NAME
REPORT_FILE = cwd + '/fpga/' + HLS_PROJECT + '/' + FPGA_PROJECT + '_prj/solution1/solution1_data.json'

#%% load fpga json report

def fpga_report(report_file, fpga_project):
    json_file = open(report_file)
    json_str = json_file.read()
    json_dict = json.loads(json_str)

    report = {}

    # Target
    report.update( {'Target': json_dict['Target']} )

    metrics = json_dict['ModuleInfo']['Metrics'][fpga_project]

    # Metrics
    report.update( metrics )

    # some calculations
    t_clk = float(report['Timing']['Target'])*1e-9
    t_clk_best = float(report['Timing']['Estimate'])*1e-9
    latency_avg = float(report['Latency']['LatencyAvg'])
    latency_worst = float(report['Latency']['LatencyWorst'])

    # Calculations
    report.update( {'Calc': {
                            'InferenceTime': latency_worst*t_clk,
                            'InferenceTimeBest': latency_avg*t_clk_best,
                            'SamplesPerSecond': 1/(latency_worst*t_clk),
                            'SamplesPerSecondBest': 1/(latency_avg*t_clk_best),
                            'ClkFreq': 1/t_clk,
                            'ClkFreqMax': 1/t_clk_best
                            }
                    } )
    
    return report

#%% print
report = fpga_report(REPORT_FILE, FPGA_PROJECT)
print(report)

# %% save report
with open(OUTPUT_FOLDER+FPGA_PROJECT+'_report.json', 'w') as f:
    json.dump(report, f, indent=4)

# %%
