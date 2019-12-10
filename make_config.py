
#%%
import os
import argparse
from soil_classifier.utils import make_config, save_config

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

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Making Config file')
    parser.add_argument('-n', '--name', metavar='model_name', type=str, required=True, help='name of the model')
    parser.add_argument('-d', '--device', type=str, help='Xilinx device part number', default='xazu7eg-fbvb900-1-i')
    parser.add_argument('-t', '--tclk', type=float, help='clock period [ns]', default=10)
    parser.add_argument('-i', '--iotype', type=str, help='input/output type: io_parallel or io_serial ', default='io_parallel')
    parser.add_argument('-p', '--precision', nargs='+', help='precision for data. Format <total bits> <mantisa>', default='io_parallel')
    parser.add_argument('-r', '--reuse', type=int, help='reuse factor', default=1)
    parser.add_argument('-td', '--test', type=str, nargs='+', help='Test data files. Format: <x_test file> <y_test file>', default=None)
    # parser.add_argument('-rf', '--root', type=str, help='root folder', default='./')

    args = parser.parse_args()

    cwd = os.getcwd()
    CONFIG_FOLDER = cwd + '/configs/'
    CONFIG_FILE = CONFIG_FOLDER + 'keras_config_{model_name}.yml'.format(model_name=args.name)

    if args.test is None:
        test_files = ['sat_x_test.dat', 'sat_y_test.dat']
    else:
        test_files = [args.test[0], args.test[1]]


    config_str = make_config(model_name=args.name,
                            part=args.device,
                            t_clk=args.tclk,
                            io_type=args.iotype,
                            precision=[args.precision[0], args.precision[1]],
                            reuse_factor=args.reuse,
                            test_data=test_files,
                            root_path=cwd)

    save_config(config_str, CONFIG_FILE)
