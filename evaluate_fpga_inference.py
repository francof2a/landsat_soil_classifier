#%%
import os
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import model_from_json

from soil_classifier.dataset import Landsat
from soil_classifier.models import  ANN50, ANN100, ANN500, \
                                    ANN100x100, ANN100x100do, ANN100x100bn,\
                                    ANN100x100x100


cwd = os.getcwd()
DATA_FOLDER = cwd + '/data/'
OUTPUT_FOLDER = cwd + '/outputs/'
MODELS_FOLDER = cwd + '/models/'
FPGA_FOLDER = cwd + '/fpga/hls_minimal/'
FPGA_INFERENCE_FILE = FPGA_FOLDER + 'tb_data/rtl_cosim_results.log'
FPGA_X_DATA_FILE = DATA_FOLDER + 'sat_x_test.dat'

SEED = 0
np.random.seed(SEED)

# %% Dataset loading
dataset = Landsat()

x_train, y_train, x_test, y_test = dataset.load(shuffle=True, seed=SEED)
num_classes = dataset.num_classes
num_bands = dataset.num_bands

# dataset standarization
x_mean = np.mean(x_train,axis=0)
x_std = np.std(x_train,axis=0)

x_mean = np.transpose(x_mean[:,np.newaxis])
x_std = np.transpose(x_std[:,np.newaxis])

# Replace zero sigma values with 1
x_std[x_std == 0] = 1

x_train_norm = np.divide( (x_train-x_mean), x_std)
x_test_norm = np.divide( (x_test-x_mean), x_std)

# labels to one hot encoding
onehotencoder = OneHotEncoder(categories='auto') 
y_train = onehotencoder.fit_transform(y_train[:,np.newaxis]).toarray()
y_test = onehotencoder.fit_transform(y_test[:,np.newaxis]).toarray() 

#%% Model
MODEL_NAME = 'ANN50x50'
metric = 'acc'
loss = 'categorical_crossentropy'
optimizer = 'nadam'

# load json and create model
model_json = open(MODELS_FOLDER+MODEL_NAME+'.json', 'r')
loaded_model_json = model_json.read()
model_json.close()

model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights(MODELS_FOLDER+MODEL_NAME+'_weights.h5')
print('Loaded model {model_name} from {model_path}'.format(model_name=MODEL_NAME, model_path=MODELS_FOLDER))
 
# evaluate loaded model on test data
model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
model.summary()

#%% inference

train_score = model.evaluate(x_train_norm, y_train, verbose=0)
print('Train score = {:.4f}'.format(train_score[1]))

test_score = model.evaluate(x_test_norm, y_test, verbose=0)
print('Test score = {:.4f}'.format(test_score[1]))

y_train_pred = model.predict(x_train_norm)
y_test_pred = model.predict(x_test_norm)

y_train_class = np.argmax(y_train, axis=1)
y_test_class = np.argmax(y_test, axis=1)
y_train_class_pred = np.argmax(y_train_pred, axis=1)
y_test_class_pred = np.argmax(y_test_pred, axis=1)

#%% iference with original model using test data for fpga
# read data used for HLS simulation
x_fpga = np.genfromtxt(FPGA_X_DATA_FILE, delimiter=' ').astype(np.float32)
# do inference with original model
y_fpga_best_pred = model.predict(x_fpga)
# convert predictions to class numbers
y_fpga_class_best_pred = np.argmax(y_fpga_best_pred, axis=1)

# accuracy
fpga_best_acc = np.mean(y_fpga_class_best_pred == y_test_class)
print('FPGA expected inference accuracy = {:.4f}'.format(fpga_best_acc))

#%% load fpga inference

# read data from HLS simulation
y_fpga_pred = np.genfromtxt(FPGA_INFERENCE_FILE, delimiter=' ')
# convert predictions to class numbers
y_fpga_class_pred = np.argmax(y_fpga_pred, axis=1)

# accuracy
fpga_acc = np.mean(y_fpga_class_pred == y_test_class)
print('FPGA inference accuracy = {:.4f}'.format(fpga_acc))


# %%
