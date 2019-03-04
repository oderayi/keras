import keras
from keras.datasets import mnist # Utility to download and load MNIST dataset
from keras.layers import Dense # Dense is a NN layer
from keras.models import Sequential # Sequential is a Neural Network of layers (Dense)
from keras.optimizers import SGD # Stochastic Gradient Descent (Loass function)
import matplotlib.pyplot as plt

# Load the mnist dataset (15mb on first run, then cached)
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# Normalize data (255 is max RGB value)
test_x = test_x.astype('float32') / 255

# Print the shapes of the data arrays
print('Test Images: ', test_x.shape)
print('Test Labels: ', test_y.shape)

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

# Load pretrained model
model.load_weights('models/mnistmodel.h5')

# Extract a specific image
img = test_x[9000]

# Flatten the image
test_img = img.reshape((1, 784))

# Predict the class
img_class = model.predict_classes(test_img)
classname = img_class[0]
print('Class: ', classname)

# Display original non-flattened image
plt.title('Prediction result: %s' %(classname))
plt.imshow(img)
plt.show()