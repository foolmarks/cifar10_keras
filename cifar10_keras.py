'''
CIFAR10 example using Keras
'''

import os
import sys
import shutil
import numpy as np
from keras import datasets, utils, layers, models, optimizers, callbacks


##############################################
# Set up directories
##############################################
# Returns the directory the current script (or interpreter) is running in
def get_script_directory():
    path = os.path.realpath(sys.argv[0])
    if os.path.isdir(path):
        return path
    else:
        return os.path.dirname(path)


SCRIPT_DIR = get_script_directory()
print('This script is located in: ', SCRIPT_DIR)

K_CHKPT_FILE = 'float-model-{epoch:02d}-{val_acc:.2f}.hdf5'
K_MODEL_DIR = os.path.join(SCRIPT_DIR, 'k_model')
K_CHKPT_DIR = os.path.join(SCRIPT_DIR, 'k_chkpts')
TB_LOG_DIR = os.path.join(SCRIPT_DIR, 'tb_logs')
K_CHKPT_PATH = os.path.join(K_CHKPT_DIR, K_CHKPT_FILE)


# create a directory for the saved model if it doesn't already exist
# delete it and recreate if it already exists
if (os.path.exists(K_MODEL_DIR)):
    shutil.rmtree(K_MODEL_DIR)
os.makedirs(K_MODEL_DIR)
print("Directory " , K_MODEL_DIR ,  "created ")


# create a directory for the TensorBoard data if it doesn't already exist
# delete it and recreate if it already exists
if (os.path.exists(TB_LOG_DIR)):
    shutil.rmtree(TB_LOG_DIR)
os.makedirs(TB_LOG_DIR)
print("Directory " , TB_LOG_DIR ,  "created ") 


# create a directory for the checkpoints if it doesn't already exist
# delete it and recreate if it already exists
if (os.path.exists(K_CHKPT_DIR)):
    shutil.rmtree(K_CHKPT_DIR)
os.makedirs(K_CHKPT_DIR)
print("Directory " , K_CHKPT_DIR ,  "created ")


#####################################################
# Hyperparameters
#####################################################
# training is unlikely to reach 250 epochs due to earlystop callback
BATCHSIZE = 50
EPOCHS = 250
LEARN_RATE = 0.0001
DECAY_RATE = 1e-6


##############################################
# Preparation of input dataset
##############################################
# CIFAR10 datset has 60k images. Training set is 50k, test set is 10k.
# Each image is 32x32 pixels RGB
(X_train, Y_train), (X_test, Y_test) = datasets.cifar10.load_data()

# Scale image data from range 0:255 to range 0:1
X_train = X_train / 255.0
X_test = X_test / 255.0

# create a list of categories (class labels)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# create unseen dataset for predictions - 5k images
X_predict = X_train[45000:]
Y_predict = Y_train[45000:]

# reduce training dataset to 45k images
X_train = X_train[:45000]
Y_train = Y_train[:45000]

# one-hot encode the labels
Y_train = utils.to_categorical(Y_train)
Y_test = utils.to_categorical(Y_test)



##############################################
# miniVGGNet as Keras functional model
##############################################
inputs = layers.Input(shape=(32, 32, 3))
net = layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
net = layers.BatchNormalization(axis=-1)(net)
net = layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(net)
net = layers.BatchNormalization(axis=-1)(net)
net = layers.MaxPooling2D(pool_size=(2,2))(net)
net = layers.Dropout(0.25)(net)
net = layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(net)
net = layers.BatchNormalization(axis=-1)(net)
net = layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(net)
net = layers.BatchNormalization(axis=-1)(net)
net = layers.MaxPooling2D(pool_size=(2,2))(net)
net = layers.Dropout(0.25)(net)
net = layers.Flatten()(net)
net = layers.Dense(512, activation='relu')(net)
net = layers.BatchNormalization()(net)
net = layers.Dropout(0.5)(net)
prediction = layers.Dense(10, activation='softmax')(net)

model = models.Model(inputs=inputs, outputs=prediction)


# print a summary of the model
print(model.summary())
print("Model Inputs: {ips}".format(ips=(model.inputs)))
print("Model Outputs: {ops}".format(ops=(model.outputs)))



##############################################
# Set up callbacks
##############################################
# create Tensorboard callback
tb_call = callbacks.TensorBoard(log_dir=TB_LOG_DIR,
                                         histogram_freq=10,
                                         batch_size=BATCHSIZE,
                                         write_graph=True,
                                         write_grads=False,
                                         write_images=False )


# Early stop callback
earlystop_call = callbacks.EarlyStopping(min_delta=0.001, patience=3)

# checkpoint save callback
chk_call = callbacks.ModelCheckpoint(K_CHKPT_PATH, save_best_only=True)



##############################################
# Compile model
##############################################
model.compile(loss='categorical_crossentropy', 
              optimizer=optimizers.Adam(lr=LEARN_RATE, decay=DECAY_RATE),
              metrics=['accuracy']
              )

##############################################
# Train model with training set
##############################################
model.fit(X_train,
          Y_train,
          batch_size=BATCHSIZE,
          shuffle=True,
          epochs=EPOCHS,
          validation_data=(X_test, Y_test),
          callbacks=[earlystop_call,tb_call,chk_call])

##############################################
# Evaluate model accuracy with test set
##############################################
scores = model.evaluate(X_test, 
                        Y_test,
                        batch_size=BATCHSIZE
                        )

print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])


##############################################
# Used trained model to make predictions
##############################################
print('Make some predictions with the trained model..')
predictions = model.predict(X_predict, batch_size=BATCHSIZE)

# each prediction is an array of 10 values
# the max of the 10 values is the model's 
# highest "confidence" classification
# use numpy argmax function to get highest of the set of 10

correct_predictions = 0
wrong_predictions = 0

for i in range(len(predictions)):
    pred=np.argmax(predictions[i])
    actual=(Y_predict[i][0])

    if (pred == actual):
        correct_predictions += 1
    else:
        wrong_predictions += 1

print('Validation dataset size: ' , len(predictions), ' Correct Predictions: ', correct_predictions, ' Wrong Predictions: ', wrong_predictions)
print ('-------------------------------------------------------------')


##############################################
# Save the model in keras format
##############################################
print("Saving the Keras model in keras format..")

# save just the weights (no architecture) to an HDF5 format file
model.save_weights(os.path.join(K_MODEL_DIR,'k_model_weights.h5'))

# save just the architecture (no weights) to a JSON file
with open(os.path.join(K_MODEL_DIR,'k_model_architecture.json'), 'w') as f:
    f.write(model.to_json())

# save weights, model architecture & optimizer to an HDF5 format file
model.save(os.path.join(K_MODEL_DIR,'k_complete_model.h5'))


print('FINISHED!')
