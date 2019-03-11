import keras
from keras.datasets import mnist # Utility to download and load MNIST dataset
from keras.layers import Dense # Dense is a NN layer
from keras.models import Sequential # Sequential is a Neural Network of layers (Dense)
from keras.optimizers import SGD # Stochastic Gradient Descent (Loass function)
from keras.callbacks import LearningRateScheduler

# Load the mnist dataset (15mb on first run, then cached)
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# Normalize data (255 is max RGB value)
train_x = train_x.astype('float32') / 255
test_x = test_x.astype('float32') / 255

# Print the shapes of the data arrays
print('Train Images: ', train_x.shape)
print('Train Labels: ', train_y.shape)
print('Test Images: ', test_x.shape)
print('Test Labels: ', test_y.shape)

# Flatten the image pixes from 2 Dimensions to 1 Dimension array of 178 rows 1 column
# (784 = 28 x 28) for faster and more efficient computation.
# Each image is 28 x 28 in dimension.
# Other algorithms like convolutional NN does not require this step
# as they can work naturally with multidimensional datasets
train_x = train_x.reshape(60000, 784)
test_x = test_x.reshape(10000, 784)

# Convert labels to vectors
# There are 10 distinct categories
# This convers labels like Dog, Cat, Bird to vector like 
# [010], [001] etc which the algorithm can work with.
train_y = keras.utils.to_categorical(train_y, 10)
test_y = keras.utils.to_categorical(test_y, 10)

# Define the model / network
## Neural network of sequential layers
model = Sequential()

## Add layers. units = neurons.
### Input layer. input_shape is necessary for Keras to figure out the input data shape
model.add(Dense(units=128, activation='relu', input_shape=(784,)))

### Other layers. input_shape will be automatically be inferred.
### First hidden layer
model.add(Dense(units=128, activation='relu'))

### Second hidden layer
model.add(Dense(units=128, activation='relu'))

### Third hidden layer
model.add(Dense(units=128, activation='relu'))

### Output layer
### Note above we have 10 image categories
### Loss function is softmax.
## Remember softmax = max(Z,0)
### Output would be an array of 10 probabilities
model.add(Dense(units=10, activation='softmax'))

# Define dynamic learning rate function (optional)
def lr_schedule(epoch):
    lr = 0.1

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

# Compile the function
# categorical_crossentropy is another name for softmax crossentropy
# metrics is the feedback we are interested in. Accuracy is important for classification problem
# which simply states the number of correctly classified items.
# Mean Squared Error should be used for regression problems instead.
model.compile(optimizer=SGD(lr_schedule(0)), loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the function (TRAIN)
# Here, we use the training dataset.
# i.e pass in the training data and train the network
# batch_size is the number of images / data we want to process at a time (i.e. mini-batch)
# epoch is the number of training iterations we want. i.e train the network on all data batches 10 times
model.fit(train_x, train_y, batch_size=32, epochs=20, shuffle=True, verbose=1, callbacks=[lr_scheduler])

# Evaluate accuracy (TEST)
# Here, we use the test dataset.
accuracy = model.evaluate(x=test_x, y=test_y, batch_size=32)
print('Accuracy: ', accuracy[1])

# Save model for use later
model.save("models/mnistmodel.h5")





