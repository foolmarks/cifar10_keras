'''
CIFAR10 example using Keras
'''

import os
import shutil
import numpy as np

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical



# Hyperparameters
BATCHSIZE = 128
EPOCHS = 20
LEARN_RATE = 0.0001
DECAY_RATE = 1e-6



##############################################
# Set up directories
##############################################

# Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath( __file__ ))
TB_LOG_DIR = os.path.join(SCRIPT_DIR, 'tb_logs')
CHKPT_DIR = os.path.join(SCRIPT_DIR, 'ckpts')
CHKPT_FILE = os.path.join(CHKPT_DIR,'model.ckpt')
MODEL_DIR = os.path.join(SCRIPT_DIR, 'model')



if (os.path.exists(TB_LOG_DIR)):
    shutil.rmtree(TB_LOG_DIR)
os.makedirs(TB_LOG_DIR)
print("Directory " , TB_LOG_DIR ,  "created ") 

if (os.path.exists(CHKPT_DIR)):
    shutil.rmtree(CHKPT_DIR)
os.makedirs(CHKPT_DIR)
print("Directory " , CHKPT_DIR ,  "created ") 

if (os.path.exists(MODEL_DIR)):
    shutil.rmtree(MODEL_DIR)
os.makedirs(MODEL_DIR)
print("Directory " , MODEL_DIR ,  "created ") 



##############################################
# Preparation of input dataset
##############################################
# CIFAR10 datset has 60k images. Training set is 50k, test set is 10k.
# Each image is 32x32 pixels RGB
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# Scale image data from range 0:255 to range 0:1
X_train = X_train / 255.0
X_test = X_test / 255.0

# create a list of categories (class labels)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


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
# Set up callbacks
##############################################
# Checkpoint callback
cp_call = ModelCheckpoint(CHKPT_FILE, 
                          save_weights_only=False,
                          period=1,
                          save_best_only=True,
                          verbose=1)

# create Tensorboard callback
tb_call = TensorBoard(log_dir=TB_LOG_DIR,
                      histogram_freq=0,
                      batch_size=BATCHSIZE,
                      write_graph=True,
                      write_grads=False,
                      write_images=False,
                      embeddings_freq=0,
                      embeddings_layer_names=None,
                      embeddings_metadata=None,
                      embeddings_data=None,
                      update_freq='epoch')

# Early stop callback
earlystop_call = EarlyStopping(min_delta=0.001, patience=3)

##############################################
# Compile model
##############################################
model.compile(loss='categorical_crossentropy', 
              optimizer=Adam(lr=LEARN_RATE,
              decay=DECAY_RATE),
              metrics=['accuracy'])

##############################################
# Train model with training set
##############################################
model.fit(X_train, to_categorical(Y_train),
          batch_size=BATCHSIZE,
          shuffle=True,
          epochs=EPOCHS,
          validation_data=(X_test, to_categorical(Y_test)),
          callbacks=[earlystop_call,cp_call,tb_call])

##############################################
# Evaluate model accuracy with test set
##############################################
scores = model.evaluate(X_test, to_categorical(Y_test))

print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])

##############################################
# Used trained model to make predictions
##############################################

print("\nLet's make some predictions with the trained model..\n")
predictions = model.predict(X_test)

# each prediction is an array of 10 values
# the max of the 10 values is the model's 
# highest "confidence" classification
# use numpy argmax function to get highest of the set of 10

print("Predict 1st sample in the test set is: {pred}".format(pred=class_names[np.argmax(predictions[0])]))
print("The 1st sample in test set actually is: {actual}".format(actual=class_names[(Y_test[0][0])]))

print("Predict 5th sample in the test set is: {pred}".format(pred=class_names[np.argmax(predictions[4])]))
print("The 5th sample in test set actually is: {actual}".format(actual=class_names[(Y_test[4][0])]))

print("Predict 9th sample in the test set is: {pred}".format(pred=class_names[np.argmax(predictions[8])]))
print("The 9th sample in test set actually is: {actual}".format(actual=class_names[(Y_test[8][0])]))


##############################################
# Save the model
##############################################
print("\nSaving the model..")


# save just the weights to an HDF5 format file
model.save_weights(os.path.join(MODEL_DIR,'model_weights.h5'))

# save the architecture to a JSON file
with open(os.path.join(MODEL_DIR,'model_architecture.json'), 'w') as f:
    f.write(model.to_json())

# save weights & architecture to an HDF5 format file
model.save(os.path.join(MODEL_DIR,'model.h5'))

print("\nFINISHED")
