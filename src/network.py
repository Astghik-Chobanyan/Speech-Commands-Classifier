import tensorflow as tf
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.models import Model

from config import config
from custom_layers import DataNormalization
from src.feature import FeatureMappings
from src.preprocessing import Preprocessing
from tensorflow.keras.layers import Flatten, Dense, Dropout
config_model = config['train_params']

"""
Here you should implement a network 
It should be LSTM or convolutional
You can implement any thing if you can reach accuracy >85% 
It should be tf.keras.Model
you are free to use ani API
"""


class Network(tf.keras.Model):
    def __init__(self, input_shape=config_model['input_shape']):
        super().__init__()
        # self.normalization = DataNormalization()
        self.dense1 = Dense(100, input_shape=input_shape, activation='relu')
        self.dropout1 = Dropout(0.5)
        self.dense2 = Dense(80, activation='relu')
        self.dropout2 = Dropout(0.5)
        self.dense3 = Dense(35, activation='softmax')

        self.inputs = Input(shape=input_shape)
        self.outputs = self.call(self.inputs)
        self.model = Model(inputs=self.inputs, outputs=self.outputs)

    def call(self, inputs):
        # x = self.normalization(inputs)
        x = self.dense1(inputs)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return self.dense3(x)


if __name__ == '__main__':
    # lstm = LSTMBasedNetwork()
    prep = Preprocessing()
    prep.create_iterators()
    train_dataset, val_dataset, test_dataset = FeatureMappings().create_features(prep)

    print('')

