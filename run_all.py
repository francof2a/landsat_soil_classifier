#%%
import os
import numpy as np 
import matplotlib.pyplot as plt 
import json

from soil_classifier.dataset import Landsat
from soil_classifier.models import minimals as models_lib
from soil_classifier.utils import fpga_report, load_model
from soil_classifier.utils import model_checkout
from soil_classifier.utils import make_config, save_config
from soil_classifier.utils import convert, build

cwd = os.getcwd()
DATA_FOLDER = cwd + '/data/'
OUTPUT_FOLDER = cwd + '/outputs/'
MODELS_FOLDER = cwd + '/models/'
CONFIG_FOLDER = cwd + '/configs/'
FPGA_FOLDER = cwd + '/fpga/'

MODEL_SRC_PATH = OUTPUT_FOLDER
MODEL_DST_PATH = MODELS_FOLDER


#%% PARAMETERS
SEED = 0

# Model
MODEL_NAME = 'ANN50x50'

# Dataset
X_DATA_PROC = 'standarization'
Y_DATA_PROC = 'one-hot'
FPGA_DATA_FORMAT = '%.6f'

# Training
N_epochs = 10
batch_size = 32
do_model_checkout = True

# Config (conversion Keras to HLS)
PART = 'xazu7eg-fbvb900-1-i'
T_CLK = 24 # ns
IO_TYPE = 'io_parallel' # options: io_serial/io_parallel
PRECISION = [24, 8]
REUSE_FACTOR = 4

# Conversion
HLS_PROJECT = 'hls_' + MODEL_NAME
FPGA_PROJECT = 'fpga_' + MODEL_NAME
CONFIG_FILE = CONFIG_FOLDER + 'keras_config_{model_name}.yml'.format(model_name=MODEL_NAME)
FPGA_PROJECT_FOLDER = FPGA_FOLDER + 'hls_' + MODEL_NAME + '/'
FPGA_INFERENCE_FILE = FPGA_PROJECT_FOLDER + 'tb_data/rtl_cosim_results.log'
OUTPUT_REPORT_FILE = OUTPUT_FOLDER + MODEL_NAME + '_report.json'

np.random.seed(SEED)

parameter_report = {'params': {
    'model_name': MODEL_NAME,
    'dataset': 'Landsat',
    'x_data_proc': X_DATA_PROC,
    'y_data_proc': Y_DATA_PROC,
    'fpga_data_format': FPGA_DATA_FORMAT,
    'epochs': N_epochs,
    'batch_size': batch_size,
    'part': PART,
    't_clk': T_CLK,
    'io_type': IO_TYPE,
    'precision': PRECISION,
    'reuse_factor': REUSE_FACTOR
}}
#%% some functions
def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

# %% Dataset loading
print('\nDataset loading and processing')
dataset = Landsat()

dataset.load(shuffle=True, seed=SEED)
x_train, y_train, x_test, y_test = dataset.posprocess(x_proc_type=X_DATA_PROC, y_proc_type=Y_DATA_PROC)

print('\nSaving data in text (dat) file for FPGA synth testing...')
DATASET_NAME = dataset.name
np.savetxt(DATA_FOLDER+DATASET_NAME+'_x_train.dat', x_train, fmt=FPGA_DATA_FORMAT)
np.savetxt(DATA_FOLDER+DATASET_NAME+'_y_train.dat', y_train, fmt=FPGA_DATA_FORMAT)
np.savetxt(DATA_FOLDER+DATASET_NAME+'_x_test.dat', x_test, fmt=FPGA_DATA_FORMAT)
np.savetxt(DATA_FOLDER+DATASET_NAME+'_y_test.dat', y_test, fmt=FPGA_DATA_FORMAT)
TEST_FILES = [DATASET_NAME+'_x_test.dat', DATASET_NAME+'_y_test.dat']
print('done!')

#%% Model
print('\nLoading and compiling model {}'.format(MODEL_NAME))
model = models_lib.new_model(MODEL_NAME)
model.compile()
model.summary()

#%% Training
print('\nTraining')
history = model.fit(x_train, y_train,
                    epochs=N_epochs, batch_size=batch_size,
                    validation_data=(x_test, y_test))


#%% evaluation
print('\n')
# show train accuracy
train_score = model.evaluate(x_train, y_train, verbose=0)
print('MODEL {} - train accuracy = {:.3f}'.format(model.name, train_score[1]))

# show test accuracy
test_score = model.evaluate(x_test, y_test, verbose=0)
print('MODEL {} - test accuracy = {:.3f}'.format(model.name, test_score[1]))

# append epochos to history
epochs = range(1,N_epochs+1)
history.history.update( {'epochs': epochs})

# save history
np.save(OUTPUT_FOLDER+'history_{}.npy'.format(model.name), history.history)
print( 'Training history saved in ' + OUTPUT_FOLDER + 'history_{}.npy'.format(model.name) )

#%% Plot training results
print('\nSaving training plots in ' + OUTPUT_FOLDER)
# loss
fig = plt.figure(figsize=(16,8))
plt.plot(epochs, history.history['loss'], label='loss')
plt.plot(epochs, history.history['val_loss'], label='val_loss')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.xlim(xmin=1)
plt.ylim(ymin=0)
plt.grid()
plt.legend()
plt.title('{} model - Loss'.format(model.name))
fig.savefig(OUTPUT_FOLDER+'{}_loss.png'.format(model.name))
# plt.show(block=False)

# acc
fig = plt.figure(figsize=(16,8))
plt.plot(epochs, history.history['acc'], label='acc')
plt.plot(epochs, history.history['val_acc'], label='val_acc')
plt.xlabel('epochs')
plt.ylabel('accuaracy')
plt.xlim(xmin=1)
plt.ylim(ymin=0.6, ymax=1.01)
plt.grid()
plt.legend()
plt.title('{} model - Accuracy'.format(model.name))
fig.savefig(OUTPUT_FOLDER+'{}_acc.png'.format(model.name))
# plt.show(block=False)

# %% save model
print('\nSaving model in ' + OUTPUT_FOLDER)
model.save(OUTPUT_FOLDER)

#%% checkout model
print('\n')
# check if you are not in a ipython notebook
if not isnotebook():
    terminal_input = input('Do you want to checkout your model (it overrides previous model with same name) y/[n]: ')
    if terminal_input == 'y':
        do_model_checkout = True
    else:
        do_model_checkout = False
        print('Model checkout denied!')

if do_model_checkout:
    print('Doing model checkout...')
    model_checkout(model.name, src_path=MODEL_SRC_PATH, dst_path=MODEL_DST_PATH)
else:
    print('Your model has not been checked out, look for it in ' + OUTPUT_FOLDER )

#%% Create config file
CONFIG_FILE = CONFIG_FOLDER + 'keras_config_{model_name}.yml'.format(model_name=model.name)

config_str = make_config(model_name=model.name,
                        part=PART,
                        t_clk=T_CLK,
                        io_type=IO_TYPE,
                        precision=[PRECISION[0], PRECISION[1]],
                        reuse_factor=REUSE_FACTOR,
                        test_data=TEST_FILES,
                        root_path=cwd)

save_config(config_str, CONFIG_FILE)

# %% Conversion and building
print('Converting from keras to HLS...')

# model conversion
print('Converting {model} according to {config}'.format(model=model.name, config=CONFIG_FILE))
convert(CONFIG_FILE)

# model building
print('Building HLS project into {prj_folder}'.format(prj_folder=FPGA_PROJECT_FOLDER))
build(FPGA_PROJECT_FOLDER)


#%% Parse FPGA report
print('\nGenerating FPGA synth report')

REPORT_FILE = FPGA_FOLDER + HLS_PROJECT + '/' + FPGA_PROJECT + '_prj/solution1/solution1_data.json'
report = fpga_report(REPORT_FILE, FPGA_PROJECT)

for k in report.keys():
    print('{}:'.format(k))
    for l in report[k]:
        print('\t{}: {}'.format(l, report[k][l]))

print('\nSaving FPGA synth report')
with open(OUTPUT_FOLDER+FPGA_PROJECT+'_report.json', 'w') as f:
    json.dump(report, f)


#%% Module reloading for inference
print('\nReloading model to obtain classification performance metrics')
if do_model_checkout:
    model = load_model(MODEL_NAME, path=MODEL_DST_PATH, verbose=1)
else:
    model = load_model(MODEL_NAME, path=MODEL_SRC_PATH, verbose=1)

#%% Inferences
print('\tPerforming original model inferences')
# convert y to class format
y_train_class = np.argmax(y_train, axis=1)
y_test_class = np.argmax(y_test, axis=1)

# original model inferences
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

y_train_class_pred = np.argmax(y_train_pred, axis=1)
y_test_class_pred = np.argmax(y_test_pred, axis=1)

print('\tPerforming FPGA implementation inference over test dataset')
# iferences with original model using test data for fpga
# read data used for HLS simulation
x_fpga = np.genfromtxt(DATA_FOLDER+TEST_FILES[0], delimiter=' ').astype(np.float32)
# do inference with original model
y_fpga_best_pred = model.predict(x_fpga)
# convert predictions to class numbers
y_fpga_class_best_pred = np.argmax(y_fpga_best_pred, axis=1)

#%% accuracy metrics
print('\nAccuracy report')
model_train_acc = np.mean(y_train_class_pred == y_train_class)
print('\tOriginal model inference train accuracy = {:.4f}'.format(model_train_acc))

model_test_acc = np.mean(y_test_class_pred == y_test_class)
print('\tOriginal model inference test accuracy = {:.4f}'.format(model_test_acc))

fpga_best_acc = np.mean(y_fpga_class_best_pred == y_test_class)
print('\tFPGA expected inference accuracy = {:.4f}'.format(fpga_best_acc))

# read data from HLS simulation
y_fpga_pred = np.genfromtxt(FPGA_INFERENCE_FILE, delimiter=' ')
# convert predictions to class numbers
y_fpga_class_pred = np.argmax(y_fpga_pred, axis=1)

# accuracy
fpga_acc = np.mean(y_fpga_class_pred == y_test_class)
print('\tFPGA inference accuracy = {:.4f}'.format(fpga_acc))

metric_report = {'acc': {
    'model_train_acc': model_train_acc,
    'model_test_acc': model_test_acc,
    'fpga_best_acc': fpga_best_acc,
    'fpga_acc': fpga_acc
}}
# %% Whole report
report.update(parameter_report)
report.update(metric_report)

# save report
with open(OUTPUT_REPORT_FILE, 'w') as f:
    json.dump(report, f, indent=4)

#%% IP checkout
# TODO

# %%
print('\n Run is complete!')