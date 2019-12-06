#%%
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import model_from_json

from ..dataset import Landsat

import json

#%%
class ClassifierModel():
    def __init__(self):
        self.name = 'minimal'
        self.model = None
        self.metrics = ['acc']
        self.optimizer = 'adam'
        self.loss = 'categorical_crossentropy'
    
    def build(self):
        pass

    def compile(self, loss=None, optimizer=None, metrics=None):
        if loss is None:
            loss = self.loss
        if optimizer is None:
            optimizer = self.optimizer
        if metrics is None:
            metrics = self.metrics

        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def summary(self):
        self.model.summary()

    def get_keras_model(self):
        return self.model

    def fit(self, x_train, y_train, epochs, batch_size=None, validation_data=None):
        history = self.model.fit(x_train, y_train,
                                epochs=epochs, batch_size=batch_size,
                                validation_data=validation_data)
        return history

    def evaluate(self, x, y, verbose=0):
        score = self.model.evaluate(x, y, verbose=verbose)
        return score

    def save(self, path='./outputs/'):
        self.model.save(path + '{}.h5'.format(self.name))
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(path + '{}.json'.format(self.name), 'w') as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights(path + '{}_weights.h5'.format(self.name))

    def load(self, model_name, path='./outputs/', verbose=0):
        try:
            # load json and create model
            model_json = open(path+model_name+'.json', 'r')
            loaded_model_json = model_json.read()
            model_json.close()

            model = model_from_json(loaded_model_json)

            # load weights into new model
            model.load_weights(path+model_name+'_weights.h5')
            if verbose > 0:
                print('Loaded model {model_name} from {model_path}'.format(model_name=model_name, model_path=path))
        except:
            print('Loading model failed!')
        
        self.model = model
        return model


class ANN50(ClassifierModel):
    def __init__(self, name='ANN50'):
        super().__init__()
        self.name = name
        self.optimizer = 'nadam'
        self.model = self.build()

    def build(self):
        dataset = Landsat()
        data_dim = dataset.data_dim
        num_classes = dataset.num_classes

        inputs = Input(shape=data_dim, name='input')

        x = Flatten(name='flatten')(inputs)
        x = Dense(50, activation='relu', name='fc1')(x)
        outputs = Dense(num_classes, activation='softmax', name='predictions')(x)

        # Model compiling
        self.model = Model(inputs, outputs, name=self.name) 

        return self.model 

class ANN50x50(ClassifierModel):
    def __init__(self, name='ANN50x50'):
        super().__init__()
        self.name = name
        self.optimizer = 'nadam'
        self.model = self.build()

    def build(self):
        dataset = Landsat()
        data_dim = dataset.data_dim
        num_classes = dataset.num_classes

        inputs = Input(shape=data_dim, name='input')

        x = Flatten(name='flatten')(inputs)
        x = Dense(50, activation='relu', name='fc1')(x)
        x = Dense(50, activation='relu', name='fc2')(x)
        outputs = Dense(num_classes, activation='softmax', name='predictions')(x)

        # Model compiling
        self.model = Model(inputs, outputs, name=self.name) 

        return self.model 

class ANN50x50th(ClassifierModel):
    def __init__(self, name='ANN50x50th'):
        super().__init__()
        self.name = name
        self.optimizer = 'nadam'
        self.model = self.build()

    def build(self):
        dataset = Landsat()
        data_dim = dataset.data_dim
        num_classes = dataset.num_classes

        inputs = Input(shape=data_dim, name='input')

        x = Flatten(name='flatten')(inputs)
        x = Dense(50, activation='tanh', name='fc1')(x)
        x = Dense(50, activation='tanh', name='fc2')(x)
        outputs = Dense(num_classes, activation='softmax', name='predictions')(x)

        # Model compiling
        self.model = Model(inputs, outputs, name=self.name) 

        return self.model 


class ANN100(ClassifierModel):
    def __init__(self, name='ANN100'):
        super().__init__()
        self.name = name
        self.optimizer = 'nadam'
        self.model = self.build()

    def build(self):
        dataset = Landsat()
        data_dim = dataset.data_dim
        num_classes = dataset.num_classes

        inputs = Input(shape=data_dim, name='input')

        x = Flatten(name='flatten')(inputs)
        x = Dense(100, activation='relu', name='fc1')(x)
        outputs = Dense(num_classes, activation='softmax', name='predictions')(x)

        # Model compiling
        self.model = Model(inputs, outputs, name=self.name) 

        return self.model 

class ANN500(ClassifierModel):
    def __init__(self, name='ANN500'):
        super().__init__()
        self.name = name
        self.optimizer = 'nadam'
        self.model = self.build()

    def build(self):
        dataset = Landsat()
        data_dim = dataset.data_dim
        num_classes = dataset.num_classes

        inputs = Input(shape=data_dim, name='input')

        x = Flatten(name='flatten')(inputs)
        x = Dense(500, activation='relu', name='fc1')(x)
        outputs = Dense(num_classes, activation='softmax', name='predictions')(x)

        # Model compiling
        self.model = Model(inputs, outputs, name=self.name) 

        return self.model 
    
class ANN100x100(ClassifierModel):
    def __init__(self, name='ANN100x100'):
        super().__init__()
        self.name = name
        self.optimizer = 'nadam'
        self.model = self.build()

    def build(self):
        dataset = Landsat()
        data_dim = dataset.data_dim
        num_classes = dataset.num_classes

        inputs = Input(shape=data_dim, name='input')

        x = Flatten(name='flatten')(inputs)
        x = Dense(100, activation='relu', name='fc1')(x)
        x = Dense(100, activation='relu', name='fc2')(x)
        outputs = Dense(num_classes, activation='softmax', name='predictions')(x)

        # Model compiling
        self.model = Model(inputs, outputs, name=self.name) 

        return self.model 

class ANN100x100do(ClassifierModel):
    def __init__(self, name='ANN100x100do'):
        super().__init__()
        self.name = name
        self.optimizer = 'nadam'
        self.model = self.build()

    def build(self):
        dataset = Landsat()
        data_dim = dataset.data_dim
        num_classes = dataset.num_classes

        inputs = Input(shape=data_dim, name='input')

        x = Flatten(name='flatten')(inputs)
        x = Dense(100, activation='relu', name='fc1')(x)
        # x = Dropout(0.1, name='do1')(x)
        x = Dense(100, activation='relu', name='fc2')(x)
        x = Dropout(0.3, name='do2')(x)
        outputs = Dense(num_classes, activation='softmax', name='predictions')(x)

        # Model compiling
        self.model = Model(inputs, outputs, name=self.name) 

        return self.model

class ANN100x100bn(ClassifierModel):
    def __init__(self, name='ANN100x100bn'):
        super().__init__()
        self.name = name
        self.optimizer = 'nadam'
        self.model = self.build()

    def build(self):
        dataset = Landsat()
        data_dim = dataset.data_dim
        num_classes = dataset.num_classes

        inputs = Input(shape=data_dim, name='input')

        x = Flatten(name='flatten')(inputs)
        x = Dense(100, activation='relu', name='fc1')(x)
        x = BatchNormalization(name='bn1')(x)
        x = Dense(100, activation='relu', name='fc2')(x)
        x = BatchNormalization(name='bn2')(x)
        outputs = Dense(num_classes, activation='softmax', name='predictions')(x)

        # Model compiling
        self.model = Model(inputs, outputs, name=self.name) 

        return self.model

class ANN100x100x100(ClassifierModel):
    def __init__(self, name='ANN100x100x100'):
        super().__init__()
        self.name = name
        self.optimizer = 'nadam'
        self.model = self.build()

    def build(self):
        dataset = Landsat()
        data_dim = dataset.data_dim
        num_classes = dataset.num_classes

        inputs = Input(shape=data_dim, name='input')

        x = Flatten(name='flatten')(inputs)
        x = Dense(100, activation='relu', name='fc1')(x)
        x = Dense(100, activation='relu', name='fc2')(x)
        x = Dense(100, activation='relu', name='fc3')(x)
        outputs = Dense(num_classes, activation='softmax', name='predictions')(x)

        # Model compiling
        self.model = Model(inputs, outputs, name=self.name) 

        return self.model 

def new_model(model_name):
    model = None
    if model_name == 'ANN50':
        model = ANN50()
    elif model_name == 'ANN50x50':
        model = ANN50x50()
    elif model_name == 'ANN50x50th':
        model = ANN50x50th()
    elif model_name == 'ANN100':
        model = ANN100()
    elif model_name == 'ANN100x100':
        model = ANN100x100()
    elif model_name == 'ANN100x100bn':
        model = ANN100x100bn()
    elif model_name == 'ANN100x100do':
        model = ANN100x100do()
    elif model_name == 'ANN100x100x100':
        model = ANN100x100x100()
    elif model_name == 'ANN500':
        model = ANN500()

    return model    