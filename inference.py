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
MODEL_NAME = 'ANN50'
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

# print(y_test[0,:], y_test_class[0])
# print(y_test_pred[0], y_test_class_pred[0])

