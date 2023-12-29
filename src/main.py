
import sys
from preprocessing import Preprocessing
from network import Network
from train import Train
from test import Test
from feature import FeatureMappings
from multiprocessing import set_start_method
from config import config

print(sys.argv)
if len(sys.argv) == 2 and sys.argv[1] == 'test':
    train_bool = False
else:
    train_bool = True

set_start_method('spawn', force=True)

prep = Preprocessing()  #creates preprocessing object
prep.create_iterators()
feature_instance = FeatureMappings() #created feature object using the prep instance
train_dataset, val_dataset, test_dataset = feature_instance.create_features(prep)  # makes datasets
network = Network()  #creates a tf.keras.Model

for i in train_dataset.take(1):
    print(i[0].shape)
    print(i[1].shape)


if train_bool:
    # runs Train.py
    train_instance = Train(network, train_dataset, val_dataset)  
    train_instance.train()  #trains the model
else:
    print('test.......')
    test_obj = Test(network, test_dataset)
    test_obj.test() #tests the model
