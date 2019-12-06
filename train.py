#%%
import os
import numpy as np 
import matplotlib.pyplot as plt 

from soil_classifier.dataset import Landsat
from soil_classifier.models import minimals as models_lib

cwd = os.getcwd()
DATA_FOLDER = cwd + '/data/'
OUTPUT_FOLDER = cwd + '/outputs/'

#%% PARAMETERS
SEED = 0

# Model
MODEL_NAME = 'ANN50x50'

# Dataset
X_DATA_PROC = 'standarization'
Y_DATA_PROC = 'one-hot'

# Training
N_epochs = 200
batch_size = 32

np.random.seed(SEED)

# %% Dataset loading
print('\nDataset loading and processing')
dataset = Landsat()

dataset.load(shuffle=True, seed=SEED)
x_train, y_train, x_test, y_test = dataset.posprocess(x_proc_type=X_DATA_PROC, y_proc_type=Y_DATA_PROC)

#%% Model
print('\nLoading and compiling model {}'.format(model.name))
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
score = model.evaluate(x_train, y_train, verbose=0)
print('MODEL {} - train accuracy = {:.3f}'.format(model.name, score[1]))

# show test accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print('MODEL {} - test accuracy = {:.3f}'.format(model.name, score[1]))

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
plt.show(block=False)

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
plt.show(block=False)

# %% save model
print('\nSaving model in ' + OUTPUT_FOLDER)
model.save(OUTPUT_FOLDER)
