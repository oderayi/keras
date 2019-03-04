import keras
from keras.datasets import mnist # Utility to download and load MNIST dataset
from keras.layers import Dense # Dense is a NN layer
from keras.models import Sequential # Sequential is a Neural Network of layers (Dense)
from keras.optimizers import SGD # Stochastic Gradient Descent (Loass function)
from keras.preprocessing import image
import matplotlib.pyplot as plt

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

# Load image from file
img = image.load_img(path='testimage.png', grayscale=True, target_size=(28, 28))
img = image.img_to_array(img)
img = img.reshape((28, 28))

# Create flattened copy of the image
test_img = img.reshape((1, 784))

# Predict the class
img_class = model.predict_classes(test_img)
classname = img_class[0]
print('Class: ', classname)

# Display original non-flattened image
plt.title('Prediction result: %s' %(classname))
plt.imshow(img)
plt.show()