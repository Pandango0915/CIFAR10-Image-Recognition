# use anaconda in terminal to create a virtual environment with python 3.7.1 and then install numpy, keras, and tensorflow in the virtual environment.
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.preprocessing import image

# Set random seed for purposes of reproducibility
seed = 21
numpy.random.seed(seed)

from keras.datasets import cifar10

# loading in the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize the inputs from 0-255 to between 0 and 1 by dividing by 255
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]

# model
model = Sequential()
# first convolutional layer
model.add(Conv2D(32, (3, 3), input_shape=(32,32,3), padding='same'))
model.add(Activation('relu')) #relu most common activation needed for layer
model.add(Dropout(0.2)) #gets rid of some data to prevent overfitting
model.add(BatchNormalization()) #normalizes all the inputs for next layer
# second convolutional layer
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
# pooling layer
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
# repeat?
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# another convolutional layer
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# another convolutional layer
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# have to flatten data
model.add(Flatten())
model.add(Dropout(0.2))
# create dense layers
model.add(Dense(256, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(128, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(class_num)) # 10 classes, each a neuron, contains probability image in class
model.add(Activation('softmax')) #takes neuron with highest probability and makes it output

epochs = 25
optimizer = 'adam' #an optimizer that comes with Keras, works good with most problems

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary())
#model.load_weights("mnist-model.h5")

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)
#saves model
model.save("mnist-model.h5")

#load image in the folder
#img = image.load_img(path="catimg.jpg",grayscale=True,target_size=(32,32,1))
#img = image.img_to_array(img)
#test_img = img.reshape((32, 32, 1))

#Model evaluation
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#use model for random image in test set
#img = test_x[130] #this is for random image from test set
#test_img = img.reshape((1,784)) #this is for random image from test set

#img_class = model.predict_classes(test_img)
#prediction = img_class[0]

#classname = img_class[0]

#print("Class: ",classname)
#img = img.reshape((32,32))
#plt.imshow(img)
#plt.title(classname)
#plt.show()

#use model for new image, have to save the model first and create a fcn that passes new image through model
#model.predict(new_image)
