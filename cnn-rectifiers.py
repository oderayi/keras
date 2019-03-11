# Convolutional neural network on CIFAR 10 with Data Augmentation
# and using alternative activations aside from ReLU like LeakyReLU, ELU (best replacement for ReLU)
# Requires GPU hardware accelleration
# For a free GPU for training, visit Google colab at https://colab.research.google.com
# To enable GPU on a notebook on Google Colab: Navigate to Edit -> Notebook Preference and choose GPU

import keras
from keras.datasets import cifar10
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, AveragePooling2D, Dropout, \
        BatchNormalization, Activation, LeakyReLU
from keras.models import Model, Input
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from math import ceil
import os


# Load the dataset (~170mb on first run, then cached)
(train_x, train_y), (test_x, test_y) = cifar10.load_data()

# Normalize data (255 is max RGB value)
train_x = train_x.astype('float32') / 255
test_x = test_x.astype('float32') / 255

# Data Augmentation:
## Subtract the mean image from both training and test sets
train_x = train_x - train_x.mean()
test_x = test_x - test_x.mean()
## Divide by the standard deviation
train_x = train_x - train_x.std(axis=0)
test_x = test_x - test_x.std(axis=0)

# Print the shapes of the data arrays
print('Train Images: ', train_x.shape)
print('Train Labels: ', train_y.shape)
print('Test Images: ', test_x.shape)
print('Test Labels: ', test_y.shape)

# Convert labels to vectors
train_y = keras.utils.to_categorical(train_y, 10)
test_y = keras.utils.to_categorical(test_y, 10)

# Convenience function to return each unit of the network
def Unit(x, filters):
    out = BatchNormalization()(x)
    out = LeakyReLU(alpha=0.25)(out)
    out = Conv2D(filters=filters, kernel_size=[3,3], strides=[1, 1], padding="same")(out)

    return out

# Define the model
## filters = feature detectors
## kernel_size = dimension of filters
## strides = steps horizontally and vertically
## padding = same padding
## pool_size = scalling factor e.g a 28 x 28 image becomes 14 x 14
def MiniModel(input_shape):
    images = Input(input_shape)

    net = Unit(images, 64)
    net = Unit(net, 64)
    net = Unit(net, 64)
    net = MaxPooling2D(pool_size=(2, 2))(net)

    net = Unit(net, 128)
    net = Unit(net, 128)
    net = Unit(net, 128)
    net = MaxPooling2D(pool_size=(2, 2))(net)

    net = Unit(net, 256)
    net = Unit(net, 256)
    net = Unit(net, 256)

    # Dropout of 0.25 means that 25% of all activations will be disabled. This is necessary to prevent overfitting.
    # Dropouts should not be overused
    net = Dropout(0.25)(net)

    # Our pooling at this point is 8 x 8 because we have had 2 pooling above
    # had scalled down our image twice by a factor of 2 per each scaling.
    # 32 / 2 = 16, 16 / 2 = 8
    # This is also a Global Average Pooling because it applies to the entire 
    # image dimension, turning every single activation map into a single scalar.
    # very useful technique.
    net = AveragePooling2D(pool_size=(8, 8))(net)
    net = Flatten()(net)
    net = Dense(units=10, activation='softmax')(net)
    
    model = Model(inputs=images, outputs=net)

    return model

# Image dimension from cifar10 is 32x32. 
# 3 is for the 3 RGB channels Red, Green, and Blue. For grascale image,
# like the ones provided by MNIST, that would be 1.
input_shape = (32, 32, 3)
model = MiniModel(input_shape)

# Print model summary
model.summary()

# Define dynamic learning rate function (optional)
def lr_schedule(epoch):
    lr = 0.001

    if epoch > 15:
        lr = lr / 100
    elif epoch > 10:
        lr = lr / 10
    elif epoch > 5:
        lr = lr / 5
    
    print('Learning rate: ', lr)

    return lr

# USe learning rate scheduer for dynamic learning rate (optional)
lr_scheduler = LearningRateScheduler(lr_schedule)

# Directory for storing out models
save_dir = os.path.join(os.getcwd(), 'cifar10savedmodels')

# Model file names
model_name = 'cifar10model.{epoch:03d}.h5'

# Create directory if it doesn't exist
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# Join the directory with the model file
model_path = os.path.join(save_dir, model_name)

# Initialize checkpoint
# period=1 means save checkpoint at every 1 epoch
checkpoint = ModelCheckpoint(filepath=model_path,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             period=1)

# Compile the function
model.compile(optimizer=Adam(lr_schedule(0)), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# More data augmentation, done by Keras backend
datagen = ImageDataGenerator(rotation_range=10, 
                             width_shift_range=5. / 32,
                             height_shift_range=5. / 32,
                             horizontal_flip=True)

# Compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(train_x)

epochs = 20
steps_per_epoch = ceil(50000/128)

# Fit the function (TRAIN)
## Using test data as validation data (split) during training
model.fit_generator(datagen.flow(train_x, train_y, batch_size=128), 
                    validation_data=[test_x, test_y],
                    epochs=epochs, steps_per_epoch=steps_per_epoch, shuffle=True, verbose=1, 
                    workers=4, callbacks=[checkpoint,lr_scheduler])

# Evaluate accuracy (TEST)
accuracy = model.evaluate(x=test_x, y=test_y, batch_size=128)
print('Accuracy: ', accuracy[1])