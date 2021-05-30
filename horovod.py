import horovod.tensorflow as hvd

hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(
                 gpus[hvd.local_rank()], 'GPU')

opt = tf.keras.optimizers.SGD(0.0005 * hvd.size())

opt = hvd.DistributedOptimizer(opt)

experimental_run_tf_function=False

callbacks = [
      hvd.callbacks.BroadcastGlobalVariablesCallback(0)
]

if hvd.rank() == 0:
   print(model.summary())

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import horovod.tensorflow.keras as hvd

import numpy as np
import argparse
import time
import sys

# sys.path.append(‘/gpfs/projects/nct00/nct00002/cifar-utils’)
# from cifar import load_cifar
# import cifar10
# cifar10.maybe_download_and_extract()


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_labels

hvd.init()
parser = argparse.ArgumentParser()
parser.add_argument(' -- epochs', type=int, default=5)
parser.add_argument(' -- batch_size', type=int, default=256)

args = parser.parse_args()
batch_size = args.batch_size
epochs = args.epochs
model_name = args.model_name

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices( gpus[hvd.local_rank()], 'GPU')
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data(batch_size)
    model = tf.keras.applications.resnet_v2.ResNet50V2(
        include_top=True, weights=None,
        input_shape=(128, 128, 3), classes=10)
    
if hvd.rank() == 0:
  print(model.summary())

opt = tf.keras.optimizers.SGD(0.0005 * hvd.size())
opt = hvd.DistributedOptimizer(opt)

model.compile(
     loss='sparse_categorical_crossentropy',
     optimizer=opt,
     metrics=['accuracy'],
     experimental_run_tf_function=False)


callbacks = [
     hvd.callbacks.BroadcastGlobalVariablesCallback(0)
]

if hvd.rank() == 0:
   verbose = 2
else:
   verbose=0

model.fit(train_images, train_labels, epochs=epochs, 
          verbose=verbose, callbacks=callbacks)
