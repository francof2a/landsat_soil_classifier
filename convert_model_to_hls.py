import os
import sys
import argparse
import hls4ml
import yaml


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



