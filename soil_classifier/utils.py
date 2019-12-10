import os
import sys
import shutil
import hls4ml
import yaml
import json
from tensorflow.keras.models import model_from_json

def model_checkout(name, src_path, dst_path):
    model_json = name + '.json'
    model_h5 = name + '.h5'
    model_weights = name + '_weights.h5'

    try:
        print('exporting {} from {} to {}'.format(name, src_path, dst_path))
        print('\tmodel json: {}'.format(model_json))
        shutil.copyfile(src_path + model_json, dst_path + model_json)
        print('\tmodel h5: {}'.format(model_h5))
        shutil.copyfile(src_path + model_h5, dst_path + model_h5)
        print('\tmodel weights: {}'.format(model_weights))
        shutil.copyfile(src_path + model_weights, dst_path + model_weights)

        print('model checkout done!')
    except:
        print('Model checkout failed!')
    return

def parse_config(config_file):
    print('Loading configuration from', config_file)
    config = open(config_file, 'r')
    return yaml.load(config, Loader=yaml.SafeLoader)

def convert(yaml_file):
    yamlConfig = parse_config(yaml_file)
    model = None

    model = hls4ml.converters.keras_to_hls(yamlConfig)

    if model is not None:
        hls4ml.writer.vivado_writer.write_hls(model)

def build(project,csim=True, synth=True, cosim=True, export=True):

    # flags conversion
    if csim:
        csim = 1
    if synth:
        synth = 1
    if cosim:
        synth = cosim = 1
    if export:
        synth = export = 1

    # Check if vivado_hls is available
    if 'linux' in sys.platform:
        found = os.system('command -v vivado_hls > /dev/null')
        if found is not 0:
            print('Vivado HLS installation not found. Make sure "vivado_hls" is on PATH.')
            sys.exit(1)
    
    # build execution
    os.system('cd {dir} && vivado_hls -f build_prj.tcl "csim={csim} synth={synth} cosim={cosim} export={export}"'.format(dir=project, csim=csim, synth=synth, cosim=cosim, export=export))

def make_config(model_name,
                part='xazu7eg-fbvb900-1-i',
                t_clk=10,
                io_type='io_parallel',
                precision=[32, 8],
                reuse_factor=1,
                strategy='Latency',
                root_path='./',
                test_data=None):
    
    '''

    test_data example: ['Landsat_x_test.dat', 'Landsat_y_test.dat']
    '''

    DATA_FOLDER = root_path + '/data/'
    OUTPUT_FOLDER = root_path + '/outputs/'
    MODELS_FOLDER = root_path + '/models/'
    CONFIG_FOLDER = root_path + '/configs/'
    FPGA_FOLDER = root_path + '/fpga/'

    config_str = 'KerasJson: {path}{model_name}.json\n'.format(path=MODELS_FOLDER, model_name=model_name)
    config_str += 'KerasH5:   {path}{model_name}_weights.h5\n'.format(path=MODELS_FOLDER, model_name=model_name)
    if test_data is not None:
        config_str += 'InputData: {path}{x_test}\n'.format(path=DATA_FOLDER, x_test=test_data[0])
        config_str += 'OutputPredictions: {path}{y_test}\n'.format(path=DATA_FOLDER, y_test=test_data[1])
    config_str += 'OutputDir: {path}hls_{model_name}\n'.format(path=FPGA_FOLDER, model_name=model_name)
    config_str += 'ProjectName: fpga_{model_name}\n'.format(model_name=model_name)
    config_str += 'XilinxPart: {part}\n'.format(part=part)
    config_str += 'ClockPeriod: {}\n'.format(t_clk)
    config_str += '\n'
    config_str += 'IOType: {}\n'.format(io_type) 
    config_str += 'HLSConfig:\n'
    config_str += '  Model:\n'
    config_str += '    Precision: ap_fixed<{},{}>\n'.format(precision[0], precision[1])
    config_str += '    ReuseFactor: {}\n'.format(reuse_factor)
    if strategy == 'Resource':
        config_str += '    Strategy: Resource\n'

    return config_str
    
def save_config(config_str, config_file):
    with open(config_file, 'w') as f:
        f.write(config_str)
    return


def ip_checkout(name, root_path=None, src_path=None, dst_path=None, version=None):
    if src_path is None:
        src_path = root_path+'/fpga/hls_'+name+'/fpga_'+name+'_prj/solution1/impl/ip/'

    if dst_path is None:
        dst_path = root_path+'/ip/'

    if root_path is None and src_path is None and dst_path is None:
        raise Exception('You must specify a path for files!')
    
    if version is None:
        version = ''

    ip_folder = name+'_ip'+version

    try:
        print('exporting IP of {} from {} to {}'.format(name, src_path, dst_path+ip_folder))
        os.system('mkdir {dst}'.format(dst=dst_path+ip_folder))
        os.system('cp -r {src}* {dst}'.format(src=src_path, dst=dst_path+ip_folder))
        print('done!')
    except:
        print('IP checkout failed!')
    return

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

def load_model(model_name, path='./outputs/', verbose=0):
    try:
        # load json and create model
        model_json = open(path+model_name+'.json', 'r')
        loaded_model_json = model_json.read()
        model_json.close()

        model = model_from_json(loaded_model_json)

        # load weights into new model
        model.load_weights(path+model_name+'_weights.h5')
        if verbose > 0:
            print('Loaded model {model_name} from {model_path}'.format(model_name=model_name, model_path=path))
    except:
        print('Loading model failed!')
    return model