import time
import argparse

import tensorflow as tf
import horovod.tensorflow.keras as hvd

from tensorflow.keras import layers, Model
from tensorflow.keras import applications
from tensorflow.keras.applications import VGG19, Xception, ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout, Flatten, Dense, BatchNormalization, UpSampling2D

# Initialize Horovod
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    
# Dataset
num_classes = 100

(X_train, y_train), (X_test, y_test) = cifar100.load_data()
Y_train = to_categorical(y_train, num_classes)
Y_test = to_categorical(y_test, num_classes)

def model(model_name, epochs, batch_size, learning_rate):
  if(model_name == 'VGG19'):
    #base_model = getattr(applications, model_name)(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dense(128))
    model.add(Dense(100, activation='softmax'))
    #model.summary()
  elif(model_name == 'Xception'):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(96, 96, 3))

    model = Sequential()
    model.add(UpSampling2D(size=(3,3)))
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dense(128))
    model.add(Dense(100, activation='softmax'))
  elif(model_name == 'ResNet50'):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dense(128))
    model.add(Dense(100, activation='softmax'))
  else:
    return print("Model not found.")

  # Horovod: adjust learning rate based on number of GPUs.
  opt = tf.optimizers.SGD(learning_rate * hvd.size(), momentum=0.9)

  # Horovod: add Horovod Distributed Optimizer.
  opt = hvd.DistributedOptimizer(opt)

  # compile the model with a SGD/momentum optimizer
  # and a very slow learning rate.
  model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'],
                experimental_run_tf_function=False) # uses hvd.DistributedOptimizer()

  # prepare data augmentation configuration
  train_datagen = ImageDataGenerator(
      rescale=1. / 255,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True)

  train_datagen.fit(X_train)
  train_generator = train_datagen.flow(X_train, Y_train, batch_size=batch_size*hvd.size())

  test_datagen = ImageDataGenerator(rescale=1. / 255)
  validation_generator = test_datagen.flow(X_test, Y_test, batch_size=batch_size*hvd.size())

  callbacks = [
      # Horovod: broadcast initial variable states from rank 0 to all other processes.
      # This is necessary to ensure consistent initialization of all workers when
      # training is started with random weights or restored from a checkpoint.
      hvd.callbacks.BroadcastGlobalVariablesCallback(0),
  ]

  # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
  #if hvd.rank() == 0:
  #    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

  # Horovod: write logs on worker 0.
  verbose = 1 if hvd.rank() == 0 else 0
  
  print("Model: {}, Epochs: {}".format(model_name, epochs))
  training_start = time.time()
  
  # fine-tune the model
  model.fit(
      train_generator,
      epochs = epochs,
      validation_data = validation_generator,
      callbacks = callbacks)
  
  training_end = (time.time() - training_start)

  print("--------------Test performance--------------")
  test_loss, test_acc = model.evaluate(X_test / 255.0, Y_test, verbose = 2)

  log_name = "Horovod_Cifar100_{}_{}.txt".format(model_name, batch_size)
  with open(log_name, "w") as f: 
    f.write("Training Time: "+ str(training_end) + "\n")
    f.write("Epochs: "+ str(epochs) + "\n")
    f.write("Test Accuracy: "+ str(test_acc) + "\n")
    
#Batch Size 256
epochs = 5
batch_size = 256
learning_rate = 1e-3

model("VGG19", epochs, batch_size, learning_rate)
model("Xception", epochs, batch_size, learning_rate)
model("ResNet50", epochs, batch_size, learning_rate)

#Batch Size 64
epochs = 5
batch_size = 64
learning_rate = 1e-3

model("VGG19", epochs, batch_size, learning_rate)
model("Xception", epochs, batch_size, learning_rate)
model("ResNet50", epochs, batch_size, learning_rate)
