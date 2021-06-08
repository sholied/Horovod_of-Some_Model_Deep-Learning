
"""# Cifar"""

import tensorflow as tf

import horovod.tensorflow as hvd
import argparse

from tensorflow.keras import datasets, layers, Model
from tensorflow.keras import applications
from tensorflow.keras.applications import VGG16, VGG19, Xception, ResNet50
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout, Flatten, Dense

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--model', type=str, default='VGG16') #VGG16/VGG19

#Not Jupyter Notebook
args = parser.parse_args()
#args = parser.parse_args([])

# Initialize Horovod
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

epochs = args.epochs
num_classes = 10

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
Y_train = to_categorical(y_train, num_classes)
Y_test = to_categorical(y_test, num_classes)

#base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
#Baru bisa VGG
base_model = getattr(applications, args.model)(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
# Extract the last layer from third block of vgg16 model
last = base_model.get_layer('block3_pool').output
# Add classification layers on top of it
x = Flatten()(last)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
pred = Dense(10, activation='sigmoid')(x)

model = Model(base_model.input, pred)

# set the base model's layers to non-trainable
# uncomment next two lines if you don't want to
# train the base model
# for layer in base_model.layers:
#     layer.trainable = False


# Horovod: adjust learning rate based on number of GPUs.
opt = tf.optimizers.SGD(args.learning_rate * hvd.size(), momentum=0.9)

# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt)

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'],
              experimental_run_tf_function=False)

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_datagen.fit(X_train)
train_generator = train_datagen.flow(X_train, Y_train, batch_size=args.batch_size*hvd.size())

test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow(X_test, Y_test, batch_size=args.batch_size*hvd.size())

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

# Horovod: write logs on worker 0.
verbose = 1 if hvd.rank() == 0 else 0

# fine-tune the model
model.fit(
    train_generator,
    epochs= epochs,
    validation_data = validation_generator,
    callbacks = callbacks)

print(args.model)
print("--------------Test performance--------------")
test_loss, test_acc = model.evaluate(X_test / 255.0, Y_test, verbose=2)

