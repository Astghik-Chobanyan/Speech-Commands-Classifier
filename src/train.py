import tensorflow as tf
import callbacks
from tensorboard.program import TensorBoard

from config import config
import os
from tensorflow.keras.metrics import CategoricalAccuracy as tf_Accuracy

config_train_params = config['train_params']


class Train:
    def __init__(self, model_object, train_dataset, dev_dataset):
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.model = model_object

        models_dir = './models'
        self.model_name = config['model_name']
       
        """
        define self.summary_dir, self.checkpoint_dir
        these are paths directories where you will store tensorboard summaries and checkpoints
        for each of your models in ./models folder should be a directory with model name, inside which should exist summaries and checpoints subdirectories
        check if they already exist and if not make them with os.makedirs        
        divide summaries into train and dev subfolders

        """
        self.summary_dir = os.path.join(models_dir, self.model_name, 'summaries')
        self.checkpoint_dir = os.path.join(models_dir, self.model_name, 'checkpoints')

        self.summary_train_dir = os.path.join(self.summary_dir, 'train')
        self.summary_dev_dir = os.path.join(self.summary_dir, 'dev')

        os.makedirs(self.summary_train_dir, exist_ok=True)
        os.makedirs(self.summary_dev_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self):
        """
        In this function 
        complile self.model object (it should be a tf.keras.Model)
        use any optimizer you like, default can be adam
        find a good loss and metric for the problem

        after that check if there exist a checkpoint, if yes: restore the model


        define callbacks

        fit the model
        """

        # Compile the model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(
            config_train_params['learning_rate']),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            print(f"Restoring from {latest_checkpoint}")
            self.model.load_weights(latest_checkpoint)

        tensorboard_callback = callbacks.TensorboardCallback(log_dir=self.summary_train_dir)

        checkpoint_filepath = os.path.join(self.checkpoint_dir,
                                           'checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5')
        checkpoint_callback = callbacks.WeightsSaver(checkpoint_dir=checkpoint_filepath)

        self.model.fit(self.train_dataset,
                       epochs=config_train_params['epochs'],
                       batch_size=config_train_params['batch_size'],
                       validation_data=self.dev_dataset,
                       callbacks=[tensorboard_callback, checkpoint_callback])
