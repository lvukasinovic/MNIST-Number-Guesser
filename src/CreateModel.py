from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


# load and format data
data = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()
train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 1).astype('float32')

# scale to 0-1 range
train_images = train_images / 255.0
test_images = test_images / 255.0

# create model
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(train_images.shape[1], train_images.shape[2], 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Trains model uses 6 epochs then saves to file
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=6)
model.save("myModel.h5")
