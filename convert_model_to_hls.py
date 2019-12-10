import os
import sys
import argparse
import hls4ml
import yaml
from soil_classifier.utils import convert, build

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Keras model to HLS and build it')
    parser.add_argument('-n', '--name', metavar='model_name', type=str, required=True, help='name of the model to convert')

    args = parser.parse_args()
    cwd = os.getcwd()

    # model convertion
    CONFIG_FOLDER = cwd + '/configs/'
    CONFIG_FILE = CONFIG_FOLDER + 'keras_config_{model_name}.yml'.format(model_name=args.name)

    convert(CONFIG_FILE)

    # model building
    FPGA_FOLDER = cwd + '/fpga/'
    FPGA_PROJECT_FOLDER = FPGA_FOLDER + 'hls_' + args.name + '/'

    build(FPGA_PROJECT_FOLDER)



