'''
CIFAR10 example using Tensorflow & Keras
'''

import os
import shutil
import numpy as np
import tensorflow as tf


print("Tensorflow version. ", tf.VERSION)
print("Keras version. ", tf.keras.__version__)


# Hyperparameters
# training is unlikely to reach 250 epochs due to earlystop callback
BATCHSIZE = 128
EPOCHS = 250
LEARN_RATE = 0.0001
DECAY_RATE = 1e-6

##############################################
# Freeze graph
# https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
##############################################
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


##############################################
# Set up directories
##############################################

# Directories
SCRIPT_DIR = os.path.dirname(os.path.abspath( __file__ ))
TB_LOG_DIR = os.path.join(SCRIPT_DIR, 'tb_logs')
MODEL_DIR = os.path.join(SCRIPT_DIR, 'model')
CHKPT_DIR = os.path.join(SCRIPT_DIR, 'chkpts')

'''
# optional code to delete existing TensorBoard logs
if (os.path.exists(TB_LOG_DIR)):
    shutil.rmtree(TB_LOG_DIR)
os.makedirs(TB_LOG_DIR)
print("Directory " , TB_LOG_DIR ,  "created ") 
'''


if (os.path.exists(MODEL_DIR)):
    shutil.rmtree(MODEL_DIR)
os.makedirs(MODEL_DIR)
print("Directory " , MODEL_DIR ,  "created ") 

if (os.path.exists(CHKPT_DIR)):
    shutil.rmtree(CHKPT_DIR)
os.makedirs(CHKPT_DIR)
print("Directory " , CHKPT_DIR ,  "created ") 



##############################################
# Preparation of input dataset
##############################################
# CIFAR10 datset has 60k images. Training set is 50k, test set is 10k.
# Each image is 32x32 pixels RGB
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()

# Scale image data from range 0:255 to range 0:1
X_train = X_train / 255.0
X_test = X_test / 255.0

# create a list of categories (class labels)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


##############################################
# Keras Sequential model
##############################################
# Create the model
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# print a summary of the model
print(model.summary())
print("Model Inputs: {ips}".format(ips=(model.inputs)))
print("Model Outputs: {ops}".format(ops=(model.outputs)))



##############################################
# Set up callbacks
##############################################
# create Tensorboard callback
tb_call = tf.keras.callbacks.TensorBoard(log_dir=TB_LOG_DIR,
                                         histogram_freq=10,
                                         batch_size=BATCHSIZE,
                                         write_graph=True,
                                         write_grads=False,
                                         write_images=False,
                                         embeddings_freq=0,
                                         embeddings_layer_names=None,
                                         embeddings_metadata=None,
                                         embeddings_data=None)

# Early stop callback
earlystop_call = tf.keras.callbacks.EarlyStopping(min_delta=0.001, patience=3)

# checkpoint save callback
chk_call = tf.keras.callbacks.ModelCheckpoint(CHKPT_DIR, save_best_only=True)



##############################################
# Compile model
##############################################
model.compile(loss='categorical_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(lr=LEARN_RATE,
              decay=DECAY_RATE),
              metrics=['accuracy'])

##############################################
# Train model with training set
##############################################
model.fit(X_train, tf.keras.utils.to_categorical(Y_train),
          batch_size=BATCHSIZE,
          shuffle=True,
          epochs=EPOCHS,
          validation_data=(X_test, tf.keras.utils.to_categorical(Y_test)),
          callbacks=[earlystop_call,tb_call,chk_call])

##############################################
# Evaluate model accuracy with test set
##############################################
scores = model.evaluate(X_test, tf.keras.utils.to_categorical(Y_test))

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

for i in range(10):
    pred=class_names[np.argmax(predictions[i])]
    actual=class_names[(Y_test[i][0])]
    print("Sample {index} in the test set is: {pred}".format(index=i, pred=pred))
    print("Sample {index} in test set actually is: {actual}".format(index=i, actual=actual))

##############################################
# Save the model in keras format
##############################################
print("\nSaving the Keras model in keras format..")

# save just the weights to an HDF5 format file
model.save_weights(os.path.join(MODEL_DIR,'k_model_weights.h5'), save_format='h5')

# save just the architecture (no weights) to a JSON file
with open(os.path.join(MODEL_DIR,'k_model_architecture.json'), 'w') as f:
    f.write(model.to_json())

# save weights, model architecture & optimizer to an HDF5 format file
model.save(os.path.join(MODEL_DIR,'k_complete_model.h5'))

##############################################
# Save the model in tensorflow format
##############################################
print("\nSaving the Keras model in TF format..")

# save just the weights to a TF checkpoint format file
model.save_weights(os.path.join(MODEL_DIR,'tf_model_weights.ckpt'), save_format='tf')



##############################################
# Freeze Keras model as TF compatible .pb file
##############################################
print("\nCreating frozen graphs and protobuf files..")


frozen_graph = freeze_session(tf.keras.backend.get_session(),
                              output_names=[out.op.name for out in model.outputs])

# Save protobuf files
tf.train.write_graph(frozen_graph, MODEL_DIR, "tf_model.pb", as_text=False)
tf.train.write_graph(frozen_graph, MODEL_DIR, "tf_model.pbtxt", as_text=True)



print("\nTensorBoard can be opened with the command:  tensorboard --logdir=./{dir} --host localhost --port 6006".format(dir=TB_LOG_DIR))

print("\nFINISHED")



