import os
import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy as tf_Accuracy
from config import config


class Test:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def test(self):
        """
        This function should do inference on test dataset
        if must be sure that checkpoints directory is not empty
        restore from the last checkpoint and perform evaluation
        print the loss and accuracy
        """
        model_name = config['model_name']
        models_dir = './models'
        checkpoint_dir = os.path.join(models_dir, model_name, 'checkpoints')
        if not os.listdir(checkpoint_dir):
            print("Checkpoint directory is empty. No model to restore.")
            return
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print(f"Restoring from {latest_checkpoint}")
            self.model.load_weights(latest_checkpoint)
        else:
            print("No checkpoint found.")
            return
        loss, accuracy = self.model.evaluate(self.dataset)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")





