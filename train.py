#%%
import os
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder 
from tensorflow.keras.optimizers import SGD

from soil_classifier.dataset import Landsat
from soil_classifier.models import  ANN50, ANN50x50, ANN100, ANN500, \
                                    ANN100x100, ANN100x100do, ANN100x100bn,\
                                    ANN100x100x100


cwd = os.getcwd()
DATA_FOLDER = cwd + '/data/'
OUTPUT_FOLDER = cwd + '/outputs/'

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
# super_model = ANN50()
super_model = ANN50x50()
# super_model = ANN100()
# super_model = ANN500()
# super_model = ANN100x100()
# super_model = ANN100x100do()
# super_model = ANN100x100bn()
# super_model = ANN100x100x100()

model = super_model.get_keras_model()

metric = 'acc'
loss = 'categorical_crossentropy'
# optimizer = 'sgd'
# optimizer = SGD(learning_rate=1e-4)
optimizer = 'nadam'

model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

model.summary()


#%% Training
N_epochs = 500
batch_size = 32

history = model.fit(x_train_norm, y_train,
                    epochs=N_epochs, batch_size=batch_size,
                    validation_data=(x_test_norm, y_test))

# show train accuracy
score = model.evaluate(x_train_norm, y_train, verbose=0)
print('MODEL {} - train accuracy = {:.3f}'.format(model.name, score[1]))

# show test accuracy
score = model.evaluate(x_test_norm, y_test, verbose=0)
print('MODEL {} - test accuracy = {:.3f}'.format(model.name, score[1]))

# save history
np.save(OUTPUT_FOLDER+'history_{}.npy'.format(model.name), history.history)

#%% Plot training results
epochs = range(1,N_epochs+1)

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
model.save(OUTPUT_FOLDER+'{}.h5'.format(model.name))
# serialize model to JSON
model_json = model.to_json()
with open(OUTPUT_FOLDER+'{}.json'.format(model.name), 'w') as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights(OUTPUT_FOLDER+'{}_weights.h5'.format(model.name))


# %%
