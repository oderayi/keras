# Convolutional neural network on CIFAR 10
# Requires GPU hardware accelleration
# For a free GPU for training, visit Google colab at https://colab.research.google.com
# To enable GPU on a notebook on Google Colab: Navigate to Edit -> Notebook Preference and choose GPU

import keras
from keras.datasets import cifar10
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, AveragePooling2D, Dropout
from keras.models import Model, Input
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import os

# Load the dataset (~170mb on first run, then cached)
(train_x, train_y), (test_x, test_y) = cifar10.load_data()

# Normalize data (255 is max RGB value)
train_x = train_x.astype('float32') / 255
test_x = test_x.astype('float32') / 255

# Print the shapes of the data arrays
print('Train Images: ', train_x.shape)
print('Train Labels: ', train_y.shape)
print('Test Images: ', test_x.shape)
print('Test Labels: ', test_y.shape)

# Convert labels to vectors
train_y = keras.utils.to_categorical(train_y, 10)
test_y = keras.utils.to_categorical(test_y, 10)

# Define the model
## filters = feature detectors
## kernel_size = dimension of filters
## strides = steps horizontally and vertically
## padding = same padding
## pool_size = scalling factor e.g a 28 x 28 image becomes 14 x 14
def MiniModel(input_shape):
    images = Input(input_shape)

    net = Conv2D(filters=64, kernel_size=[3,3], strides=[1, 1], padding="same", activation="relu")(images)
    net = Conv2D(filters=64, kernel_size=[3,3], strides=[1, 1], padding="same", activation="relu")(net)
    net = Conv2D(filters=64, kernel_size=[3,3], strides=[1, 1], padding="same", activation="relu")(net)
    net = MaxPooling2D(pool_size=(2, 2))(net)
    net = Conv2D(filters=128, kernel_size=[3,3], strides=[1, 1], padding="same", activation="relu")(net)
    net = MaxPooling2D(pool_size=(2, 2))(net)
    net = Conv2D(filters=256, kernel_size=[3,3], strides=[1, 1], padding="same", activation="relu")(net)
    net = Conv2D(filters=256, kernel_size=[3,3], strides=[1, 1], padding="same", activation="relu")(net)
    net = Conv2D(filters=256, kernel_size=[3,3], strides=[1, 1], padding="same", activation="relu")(net)

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

# Fit the function (TRAIN)
## Using test data as validation data (split) during training
model.fit(train_x, train_y, batch_size=128, epochs=20, shuffle=True, 
          validation_split=0.1, verbose=1, 
          callbacks=[checkpoint,lr_scheduler])

# Evaluate accuracy (TEST)
accuracy = model.evaluate(x=test_x, y=test_y, batch_size=128)
print('Accuracy: ', accuracy[1])