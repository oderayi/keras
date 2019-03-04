import keras
from keras.datasets import mnist # Utility to download and load MNIST dataset
from keras.layers import Dense # Dense is a NN layer
from keras.models import Sequential # Sequential is a Neural Network of layers (Dense)
from keras.optimizers import SGD # Stochastic Gradient Descent (Loass function)

# Load the mnist dataset (15mb on first run, then cached)
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# Normalize data (255 is max RGB value)
test_x = test_x.astype('float32') / 255

# Print the shapes of the data arrays
print('Test Images: ', test_x.shape)
print('Test Labels: ', test_y.shape)

# Flatten the image pixes from 2 Dimensions to 1 Dimension array of 178 rows 1 column
test_x = test_x.reshape(10000, 784)

# Convert labels to vectors
test_y = keras.utils.to_categorical(test_y, 10)

# Define the model / network
model = Sequential()
model.add(Dense(units=128, activation='relu', input_shape=(784,)))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

# Compile the function
model.compile(optimizer=SGD(0.01), loss='categorical_crossentropy', metrics=['accuracy'])

# Load saved model
model.load_weights('models/mnistmodel.h5')

# Evaluate accuracy (TEST)
accuracy = model.evaluate(x=test_x, y=test_y, batch_size=32)
print('Accuracy: ', accuracy[1])





