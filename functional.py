import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Model, Input
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import os

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

# Flatten the image pixes
train_x = train_x.reshape(60000, 784)
test_x = test_x.reshape(10000, 784)

# Convert labels to vectors
train_y = keras.utils.to_categorical(train_y, 10)
test_y = keras.utils.to_categorical(test_y, 10)

# Define the model
def MiniModel(input_shape):
    images = Input(input_shape)

    net = Dense(units=128, activation='relu')(images)
    net = Dense(units=128, activation='relu')(net)
    net = Dense(units=128, activation='relu')(net)
    net = Dense(units=10, activation='softmax')(net)
    
    model = Model(inputs=images, outputs=net)

    return model

model = MiniModel((784,))

# Print model summary
model.summary()

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

# Directory for storing out models
save_dir = os.path.join(os.getcwd(), 'mnistsavedmodels')

# Model file names
model_name = 'mnistmodel.{epoch:03d}.h5'

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
                             period=1)

# Compile the function
model.compile(optimizer=SGD(lr_schedule(0)), loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the function (TRAIN)
## Using test data as validation data (split) during training
model.fit(train_x, train_y, batch_size=32, epochs=20, shuffle=True, validation_data=[test_x, test_y],
    verbose=1, callbacks=[checkpoint,lr_scheduler])

# Evaluate accuracy (TEST)
accuracy = model.evaluate(x=test_x, y=test_y, batch_size=32)
print('Accuracy: ', accuracy[1])