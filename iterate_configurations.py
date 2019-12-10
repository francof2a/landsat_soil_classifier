#%%
import os
import numpy as np 
import sys
sys.path.append("..")

from soil_classifier.utils import make_config, save_config


cwd = os.getcwd() + '/../'
DATA_FOLDER = cwd + '/data/'
OUTPUT_FOLDER = cwd + '/outputs/'
MODELS_FOLDER = cwd + '/models/'
CONFIG_FOLDER = cwd + '/configs/'
FPGA_FOLDER = cwd + '/fpga/'
IPS_FOLDER = cwd + '/ip/'

#%% PARAMETERS
# Dataset
DATASET_NAME = 'Landsat'
TEST_FILES = [DATASET_NAME+'_x_test.dat', DATASET_NAME+'_y_test.dat']

# Model
MODEL_NAME = 'ANN50x50'

# Config (conversion Keras to HLS)
PART = 'xazu7eg-fbvb900-1-i'
T_CLK = 24 # ns
IO_TYPE = 'io_parallel' # options: io_serial/io_parallel

# This parameters will going to be iterate to create configurations
# configuration file name format: keras_config_<MODEL_NAME>_p<PRECISION>_r<REUSE_FACTOR>
# Parameters should be specified as lists
PRECISION = [[32, 8], [24,8], [16,8], [32,6], [24,6], [16,6]]
REUSE_FACTOR = [1,2,3,4,8,12,16]



#%% Create config file
for p in PRECISION:
    for r in REUSE_FACTOR:
        config_str = make_config(model_name=MODEL_NAME,
                        part=PART,
                        t_clk=T_CLK,
                        io_type=IO_TYPE,
                        precision=[p[0], p[1]],
                        reuse_factor=r,
                        test_data=TEST_FILES,
                        root_path=cwd)

        CONFIG_FILE = CONFIG_FOLDER + 'keras_config_{model_name}_p{p0}_{p1}_r{r}.yml'.format(
            model_name=MODEL_NAME,
            p0=p[0],
            p1=p[1],
            r=r
        )                
        save_config(config_str, CONFIG_FILE)
