import time
import argparse
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout, Flatten, Dense, UpSampling2D
import horovod.tensorflow.keras as hvd

#clear cache
tf.keras.backend.clear_session()

#Training settings
parser = argparse.ArgumentParser(description='Tensorflow Horovod')
parser.add_argument('-dataset', type=str, default="cifar10", metavar='cifar10', help="Datasets cifar10 dan cifar100")
parser.add_argument('-batchsize', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('-epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 5)')
parser.add_argument('-lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
args = parser.parse_args()


print("Dataset =", args.dataset)
print("batch size =", args.batchsize)
print("learning rate =", args.lr)
print("epoch =", args.epochs)

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
num_classes = 10

if (args.dataset == "cifar10"):
  (X_train, y_train), (X_test, y_test) = cifar10.load_data()
elif (args.dataset == "cifar100"):
  (X_train, y_train), (X_test, y_test) = cifar100.load_data()
else:
  print("Dataset not found.")

Y_train = to_categorical(y_train, num_classes)
Y_test = to_categorical(y_test, num_classes)

model_name = 'Xception'
epochs = args.epochs
batch_size = args.batchsize
learning_rate = args.lr

base_model = Xception(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
model = Sequential()
model.add(UpSampling2D(size=(3,3)))
model.add(base_model)
model.add(Flatten())
model.add(Dense(256))
model.add(Dense(64))
model.add(Dense(10, activation='softmax'))

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

log_name = "Horovod_{}_{}_{}.txt".format(args.dataset, model_name, batch_size)
with open(log_name, "w") as f: 
  f.write("Training Time: "+ str(training_end) + "\n")
  f.write("Model: "+ str(model_name) + "\n")
  f.write("Epochs: "+ str(epochs) + "\n")
  f.write("Batch Size: "+ str(batch_size) + "\n")
  f.write("lerning rate: "+ str(learning_rate) + "\n")
  f.write("Dataset: "+ str(args.dataset) + "\n")
  f.write("Test Accuracy: "+ str(test_acc) + "\n")