import os

import tensorflow as tf
from config import config

train_config = config['train_params']


class TensorboardCallback(tf.keras.callbacks.Callback):
    """
    this class is for tensorboard summaries
    implement __init__() and on_train_batch_end() functions
    you should be able to save summaries with config['train_params']['summary_step'] frequency
    tensorboard should show loss and accuracy for train and validation separately
    those in their respective folders defined in train
    """
    def __init__(self, log_dir):
        super().__init__()
        self.summary_step = train_config['summary_step']
        self.train_log_dir = log_dir + '/train'
        self.val_log_dir = log_dir + '/val'
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(self.val_log_dir)
        self.step = 0

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.step += 1
        if self.step % self.summary_step == 0:
            with self.train_summary_writer.as_default():
                for key, value in logs.items():
                    if 'loss' in key or 'accuracy' in key:
                        tf.summary.scalar(key, value, step=self.step)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with self.val_summary_writer.as_default():
            for key, value in logs.items():
                if 'val_loss' in key or 'val_accuracy' in key:
                    tf.summary.scalar(key, value, step=self.step)


class WeightsSaver(tf.keras.callbacks.Callback):
    """
    this class is for checkpoints
    implement __init__ and on_train_batch_end functions and any other auxilary functions you may need
    it should be able to save at  config['train_params']['latest_checkpoint_step']
    it should save 'max_to_keep' number of checkpoints EX. if max_to_keep = 5, you should keep only 5 newest checkpoints
    save in the folder defined in train 
    """
    def __init__(self, checkpoint_dir):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.latest_checkpoint_step = train_config['latest_checkpoint_step']
        self.max_to_keep = train_config['max_checkpoints_to_keep']
        self.step = 0
        self.checkpoints = []

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step % self.latest_checkpoint_step == 0:
            self._save_checkpoint()

    def _save_checkpoint(self):
        checkpoint_file = os.path.join(self.checkpoint_dir, f'ckpt_{self.step}')
        self.model.save_weights(checkpoint_file)
        self.checkpoints.append(checkpoint_file)

        # Keep only the max_to_keep most recent checkpoints
        if len(self.checkpoints) > self.max_to_keep:
            oldest_checkpoint = self.checkpoints.pop(0)
            os.remove(oldest_checkpoint)

