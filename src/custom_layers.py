import tensorflow as tf


class DataNormalization(tf.keras.layers.Layer):
    # implement __init__(), call, build functions for this layer
    # it should normalize the data with using mean and variance,
    # which have to be updated for every train step
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean = self.add_weight(name='mean', shape=(1,), initializer='zeros',
                                    trainable=False)
        self.variance = self.add_weight(name='variance', shape=(1,), initializer='ones',
                                        trainable=False)

    def build(self, input_shape):
        super(DataNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        if training:
            batch_mean, batch_variance = tf.nn.moments(inputs, axes=[0], keepdims=True)
            self.mean.assign(batch_mean)
            self.variance.assign(
                batch_variance + 1e-6)

        return (inputs - self.mean) / tf.sqrt(self.variance)

    def compute_output_shape(self, input_shape):
        return input_shape

