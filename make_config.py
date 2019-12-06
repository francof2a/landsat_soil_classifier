
#%%
import os
import argparse

model_name = 'ANN50x50'
part = 'xazu7eg-fbvb900-1-i'
t_clk = 24 # ns
io_type = 'io_parallel' # options: io_serial/io_parallel
precision = [24, 8]
reuse_factor = 4

cwd = os.getcwd()
DATA_FOLDER = cwd + '/data/'
OUTPUT_FOLDER = cwd + '/outputs/'
MODELS_FOLDER = cwd + '/models/'
CONFIG_FOLDER = cwd + '/configs/'
FPGA_FOLDER = cwd + '/fpga/'

CONFIG_FILE = CONFIG_FOLDER + 'keras_config_{model_name}.yml'.format(model_name=model_name)
#%%

def make_config(model_name,
                part='xazu7eg-fbvb900-1-i',
                t_clk=10,
                io_type='io_parallel',
                precision=[32, 8],
                reuse_factor=1,
                root_path='./',
                test_data=None):
    
    '''

    test_data example: ['sat_x_test.dat', 'sat_y_test.dat']
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
    config_str += 'OutputDir: {path}hls\n'.format(path=FPGA_FOLDER)
    config_str += 'ProjectName: fpga_{model_name}\n'.format(model_name=model_name)
    config_str += 'XilinxPart: {part}\n'.format(part=part)
    config_str += 'ClockPeriod: {}\n'.format(t_clk)
    config_str += '\n'
    config_str += 'IOType: {}\n'.format(io_type) 
    config_str += 'HLSConfig:\n'
    config_str += '  Model:\n'
    config_str += '    Precision: ap_fixed<{},{}>\n'.format(precision[0], precision[1])
    config_str += '    ReuseFactor: {}\n'.format(reuse_factor)

    return config_str
    
def save_config(config_str, config_file):
    with open(config_file, 'w') as f:
        f.write(config_str)
    return

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Making Config file')
    parser.add_argument('-n', '--name', metavar='model_name', type=str, required=True, help='name of the model to checkout')

    args = parser.parse_args()

    cwd = os.getcwd()
    CONFIG_FOLDER = cwd + '/configs/'
    CONFIG_FILE = CONFIG_FOLDER + 'keras_config_{model_name}.yml'.format(model_name=args.name)

    test_files = ['sat_x_test.dat', 'sat_y_test.dat']

    config_str = make_config(model_name=args.name,
                            test_data=test_files,
                            root_path=cwd)
    save_config(config_str, CONFIG_FILE)
