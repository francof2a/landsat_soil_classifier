#%%
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization

from ..dataset import Landsat

#%%
class ClassifierModel():
    def __init__(self):
        self.name = 'minimal'
        self.model = None
    
    def build(self):
        pass

    def get_keras_model(self):
        return self.model

class ANN100(ClassifierModel):
    def __init__(self, name='ANN100'):
        super().__init__()
        self.name = name
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