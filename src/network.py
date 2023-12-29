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
        self.model = Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),

            layers.Reshape((-1, 64)),  
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.5),

            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),

            layers.Dense(35, activation='softmax')
        ])
        
    def call(self, inputs):
        # x = self.normalization(inputs)
        return self.model(inputs)
           


if __name__ == '__main__':
    # lstm = LSTMBasedNetwork()
    prep = Preprocessing()
    prep.create_iterators()
    train_dataset, val_dataset, test_dataset = FeatureMappings().create_features(prep)


