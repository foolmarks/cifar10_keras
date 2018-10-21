'''
CIFAR10 example using KEras & TensorFlow
'''

import numpy as np

from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical


##############################################
# Preparation of input dataset
##############################################
# CIFAR10 datset has 60k images. Trainge set is 50k, test set is 10k.
# Each image is 32x32 pixels RGB
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()


##############################################
# Keras Sequential model
##############################################
# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# print a summary of the model
print(model.summary())

##############################################
# Compile model
##############################################
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

##############################################
# Train model with training set
##############################################
model.fit(X_train / 255.0, to_categorical(Y_train),
          batch_size=128,
          shuffle=True,
          epochs=5,
          validation_data=(X_test / 255.0, to_categorical(Y_test)),
          callbacks=[EarlyStopping(min_delta=0.001, patience=3)])

##############################################
# Evaluate model accuracy with test set
##############################################
scores = model.evaluate(X_test / 255.0, to_categorical(Y_test))

print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])
